package main

import (
	"fmt"
	"image"
	"sync"

	"github.com/hkloudou/cvmatch"
)

// The engine below is the reference wiring of the Matcher/Fleet API for
// the window-automation shape: one rule set per window (a matches table
// keyed by window_id — different windows carry different logic), each
// rule watching an ordered list of templates inside an optional ROI,
// with a threshold and a match mode. Build one Engine per window at
// rule-load time; call Run once per captured frame. Everything heavy is
// prepared once and shared:
//
//   - every unique (template id, gray) pair becomes ONE Matcher no
//     matter how many rules reference it (pixels, exact statistics);
//   - rules are grouped by (ROI, gray) and each group is ONE Fleet, so
//     the ROI view's conversion, alpha scan and forward FFT are paid
//     once per frame for the whole group — rules without an ROI all
//     land in the shared whole-window group, the biggest win;
//   - one Matcher may sit in several fleets at once: a fleet keeps the
//     member spectra in its own per-geometry cache (it never touches
//     the Matcher's solo-Find cache), and for a stable window size
//     every cache hits from the second frame on.
//
// Policy stays out of the library on purpose: thresholds, the two match
// modes, coordinate mapping and oversized-template handling are the few
// lines of Run below. ROI search needs no dedicated API — SubImage
// views are zero-copy and cut the search cost to the ROI area itself,
// which no amount of spectrum sharing can beat (see the fleet-roi
// verdict in CLAUDE.md for why per-member ROIs inside one full-frame
// fleet were parked).

// Template is one watched image, decoded once at rule-load time (in a
// real service: SELECT blob_data → image.Decode → Template). Reusing an
// ID for different pixel data is caller error — the first image wins.
type Template struct {
	ID  int64
	Img image.Image
}

// Rule mirrors one row of a matches table: watch Templates inside ROI
// (empty = whole window) and report a verdict per Mode against
// Threshold.
type Rule struct {
	ID   int64
	Name string
	// ROI is in window coordinates ((0,0) = window top-left); the empty
	// rectangle means the whole window. Relative/percent ROI specs
	// should be resolved to pixels before NewEngine.
	ROI image.Rectangle
	// Gray matches on grayscale — a few times cheaper and right for
	// most UI hunting; keep color for states that differ mainly in
	// saturation (enabled vs grayed-out icons).
	Gray      bool
	Threshold float32
	Mode      int // ModeFirst or ModeAll
	Templates []Template
}

const (
	ModeFirst = 0 // templates tried in order, first one passing wins
	ModeAll   = 1 // every template evaluated, best passing one wins
)

// Hit is one template located in the frame, in window coordinates —
// add the window's screen origin for screen coordinates.
type Hit struct {
	TemplateID int64
	Score      float32
	Rect       image.Rectangle
}

// RuleResult is one rule's verdict for one frame. Templates that were
// not evaluated (larger than their resolved ROI, or an ROI outside the
// frame) simply do not appear: where one-shot Match would panic, the
// engine degrades to "not matched".
type RuleResult struct {
	RuleID  int64
	Name    string
	Matched bool
	Best    *Hit
	All     []Hit // ModeAll only: every evaluated template's best position
}

type ref struct{ rule, tpl int }

// member is one deduped matcher inside a group plus every (rule,
// template) slot its single fleet result answers.
type member struct {
	m    *cvmatch.Matcher
	w, h int
	refs []ref
}

// group is one (ROI, gray) class sharing a single Fleet. sub is the
// fallback fleet for view sizes not every member fits — a single slot
// keyed by the last seen size, rebuilt only when it changes, exactly
// like the library's own geometry caches (a window resize costs one
// rebuild, a stable stream never rebuilds).
type group struct {
	roi     image.Rectangle
	gray    bool
	members []member
	fleet   *cvmatch.Fleet
	allIdx  []int

	mu         sync.Mutex
	subW, subH int
	sub        *cvmatch.Fleet
	subIdx     []int
}

// Engine is one window's compiled rule set. Safe for concurrent Run.
type Engine struct {
	rules  []Rule
	groups []*group
}

// NewEngine compiles a window's rules: one Matcher per unique
// (template, gray) pair, one Fleet per (ROI, gray) group. Plans and
// spectra build lazily inside the fleets on the first Run per view
// geometry, so construction is cheap (pixel copies + exact statistics).
func NewEngine(rules []Rule) (*Engine, error) {
	e := &Engine{rules: rules}
	type mkey struct {
		id   int64
		gray bool
	}
	type gkey struct {
		roi  image.Rectangle
		gray bool
	}
	matchers := map[mkey]*cvmatch.Matcher{}
	groups := map[gkey]*group{}
	for ri := range rules {
		r := &rules[ri]
		if r.Mode != ModeFirst && r.Mode != ModeAll {
			return nil, fmt.Errorf("windowrules: rule %d: unknown mode %d", r.ID, r.Mode)
		}
		for ti := range r.Templates {
			t := &r.Templates[ti]
			if t.Img == nil {
				return nil, fmt.Errorf("windowrules: rule %d: template %d has no image", r.ID, t.ID)
			}
			b := t.Img.Bounds()
			if b.Dx() <= 0 || b.Dy() <= 0 {
				return nil, fmt.Errorf("windowrules: rule %d: template %d is empty", r.ID, t.ID)
			}
			mk := mkey{t.ID, r.Gray}
			m := matchers[mk]
			if m == nil {
				if r.Gray {
					m = cvmatch.NewGrayMatcher(t.Img)
				} else {
					m = cvmatch.NewMatcher(t.Img)
				}
				matchers[mk] = m
			}
			gk := gkey{canonROI(r.ROI), r.Gray}
			g := groups[gk]
			if g == nil {
				g = &group{roi: gk.roi, gray: r.Gray}
				groups[gk] = g
				e.groups = append(e.groups, g)
			}
			mi := -1
			for i := range g.members {
				if g.members[i].m == m {
					mi = i
					break
				}
			}
			if mi < 0 {
				g.members = append(g.members, member{m: m, w: b.Dx(), h: b.Dy()})
				mi = len(g.members) - 1
			}
			g.members[mi].refs = append(g.members[mi].refs, ref{ri, ti})
		}
	}
	for _, g := range e.groups {
		ms := make([]*cvmatch.Matcher, len(g.members))
		g.allIdx = make([]int, len(g.members))
		for i := range g.members {
			ms[i], g.allIdx[i] = g.members[i].m, i
		}
		g.fleet = cvmatch.NewFleet(ms...)
	}
	return e, nil
}

// canonROI maps every "no ROI" spelling to the zero rectangle so all
// whole-window rules share one group key.
func canonROI(r image.Rectangle) image.Rectangle {
	if r.Empty() {
		return image.Rectangle{}
	}
	return r.Canon()
}

type subImager interface {
	SubImage(image.Rectangle) image.Image
}

type outcome struct {
	ok    bool
	score float32
	pos   image.Point
}

// Run matches one captured frame against every rule and returns the
// verdicts in rule order. The frame's Bounds().Min is treated as the
// window origin (screen captures have Min == (0,0)).
func (e *Engine) Run(frame image.Image) ([]RuleResult, error) {
	fb := frame.Bounds()
	out := make([][]outcome, len(e.rules))
	for ri := range e.rules {
		out[ri] = make([]outcome, len(e.rules[ri].Templates))
	}
	for _, g := range e.groups {
		rect := fb
		if !g.roi.Empty() {
			rect = g.roi.Add(fb.Min).Intersect(fb)
			if rect.Empty() {
				continue // ROI outside the frame: its templates stay unevaluated
			}
		}
		view := frame
		if rect != fb {
			si, ok := frame.(subImager)
			if !ok {
				return nil, fmt.Errorf("windowrules: frame type %T lacks SubImage (needed for ROI rules)", frame)
			}
			view = si.SubImage(rect) // zero-copy for *image.RGBA / *image.Gray
		}
		fleet, idx := g.fit(rect.Dx(), rect.Dy())
		if fleet == nil {
			continue // nothing fits this ROI at this frame size
		}
		for i, r := range fleet.FindAll(view) {
			mem := &g.members[idx[i]]
			pos := rect.Min.Add(image.Pt(r.MaxX, r.MaxY)).Sub(fb.Min)
			for _, rf := range mem.refs {
				out[rf.rule][rf.tpl] = outcome{ok: true, score: r.MaxV, pos: pos}
			}
		}
	}
	results := make([]RuleResult, len(e.rules))
	for ri := range e.rules {
		r := &e.rules[ri]
		rr := RuleResult{RuleID: r.ID, Name: r.Name}
		for ti := range r.Templates {
			o := out[ri][ti]
			if !o.ok {
				continue
			}
			b := r.Templates[ti].Img.Bounds()
			hit := Hit{TemplateID: r.Templates[ti].ID, Score: o.score,
				Rect: image.Rectangle{Min: o.pos, Max: o.pos.Add(image.Pt(b.Dx(), b.Dy()))}}
			pass := o.score >= r.Threshold
			if r.Mode == ModeFirst {
				// Ordered first-hit semantics are kept on the SELECTION;
				// the compute is batched anyway because within a shared
				// fleet every extra member costs only its tail — far less
				// than a serial one-shot Match per template.
				if pass {
					rr.Matched, rr.Best = true, &hit
					break
				}
				continue
			}
			rr.All = append(rr.All, hit)
			// Strictly-greater keeps the first occurrence on ties.
			if pass && (rr.Best == nil || hit.Score > rr.Best.Score) {
				h := hit
				rr.Matched, rr.Best = true, &h
			}
		}
		results[ri] = rr
	}
	return results, nil
}

// fit returns the fleet to run against a w×h view: the full fleet when
// every member fits, otherwise a memoized sub-fleet of the members that
// do (nil when none does).
func (g *group) fit(w, h int) (*cvmatch.Fleet, []int) {
	fits := 0
	for i := range g.members {
		if g.members[i].w <= w && g.members[i].h <= h {
			fits++
		}
	}
	if fits == len(g.members) {
		return g.fleet, g.allIdx
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.subW != w || g.subH != h {
		g.subW, g.subH, g.sub, g.subIdx = w, h, nil, nil
		if fits > 0 {
			ms := make([]*cvmatch.Matcher, 0, fits)
			idx := make([]int, 0, fits)
			for i := range g.members {
				if g.members[i].w <= w && g.members[i].h <= h {
					ms = append(ms, g.members[i].m)
					idx = append(idx, i)
				}
			}
			g.sub, g.subIdx = cvmatch.NewFleet(ms...), idx
		}
	}
	return g.sub, g.subIdx
}
