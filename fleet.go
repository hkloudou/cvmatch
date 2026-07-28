package cvmatch

import (
	"fmt"
	"image"
	"sort"
	"sync"
	"sync/atomic"
)

// Result is one matcher's answer from Fleet.FindAll — Match's six-tuple
// as a struct, in the fleet's construction order.
type Result struct {
	MinV       float32
	MinX, MinY int
	MaxV       float32
	MaxX, MaxY int
}

// Fleet batches many prepared Matchers against one parent frame — the
// production shape where hundreds of templates are hunted across every
// screenshot. One FindAll converts the parent once, scans its alpha
// once, and computes the parent-side forward FFT once per geometry
// bucket into a pooled frame-spectrum cache; each matcher then pays
// only its own tail (conjugate multiply, inverse transform, emit,
// normalization, extremum scan).
//
// Members are bucketed by shape dominance: a bucket's geometry is
// always an actual member's size class, never a synthesis of unrelated
// maxima (a 1500×10 and a 10×1500 member get separate buckets instead
// of a fictitious 1500×1500 plan). Within a bucket everyone shares one
// tile geometry planned for its largest member, so per-member results
// can deviate from solo Find at the usual plan-change tolerance — the
// same contract as the tile argmin itself, ~1e-5 score class for
// well-conditioned templates, up to the parity element class (~1e-3)
// for near-flat, low-variance ones whose tiny norms amplify FFT
// rounding. Two consequences, exactly as with any plan change: on
// EXACT repeats of a template the mathematically tied maxima are
// separated only by rounding noise, so the fleet may report a
// different — equally valid — occurrence than solo Find (within one
// fleet the choice is deterministic); and confidence thresholds
// should not sit within ~1e-3 of a decision boundary for degenerate
// templates. A single-member fleet plans identically to Find and
// reproduces it bit for bit. For a fixed parent size every bucket's
// plan and every member's spectrum build once, on the first frame.
//
// FindAll is safe for concurrent use. Frame-cache memory during a call
// is cnMax·Σ_tiles·specN complex64s per bucket, pooled and released as
// each bucket completes; each member's cached spectrum adds cn·specN
// as with a solo Matcher.
type Fleet struct {
	ms      []*Matcher
	step    int
	buckets []fleetBucket
	mu      sync.Mutex
	geom    atomic.Pointer[fleetGeom]
}

// fleetBucket groups non-flat members whose pow2 size class is
// dominated by the bucket's own class; tw/th are the actual maxima of
// its members, so the plan is always shaped like a real template.
type fleetBucket struct {
	tw, th  int
	members []int // global member indices
}

type fleetGeom struct {
	pw, ph int
	pConst bool  // parent alpha constant (color mode)
	cns    []int // per member; 0 for flat members
	bg     []bucketGeom
}

type bucketGeom struct {
	p     *goPlan // shared tile geometry (bucket tw, th)
	p2    *goPlan // shrunk last-band plan, nil when the band keeps dftH
	sets  []*tspecSet
	order []int // positions into members, costliest tail first
	cnMax int
}

// NewFleet groups prepared Matchers for batched searching. All members
// must share one mode (all NewMatcher or all NewGrayMatcher); panics on
// an empty or mixed fleet.
func NewFleet(ms ...*Matcher) *Fleet {
	if len(ms) == 0 {
		panic("cvmatch: empty fleet")
	}
	// Private copy: a caller spreading its own slice could otherwise
	// swap members after construction, desyncing the cached geometry
	// (codex finding, PR #34).
	f := &Fleet{ms: append([]*Matcher(nil), ms...), step: ms[0].step}
	for _, m := range f.ms {
		if m.step != f.step {
			panic("cvmatch: fleet mixes color and gray matchers")
		}
	}
	f.buckets = bucketize(f.ms)
	return f
}

// bucketize groups the non-flat members (flat templates answer
// constant 1 without touching the pipeline, so they never shape a
// plan — codex finding, PR #34). Classes are keyed by pow2-rounded
// dimensions and merged only under shape dominance, largest area
// first: a class joins the first bucket that covers it in BOTH axes,
// so bucket dimensions are always a real member's class and orthogonal
// extremes (wide×short vs tall×narrow) stay apart instead of
// synthesizing a giant virtual template (codex finding, PR #34).
// Deterministic and independent of member order.
func bucketize(ms []*Matcher) []fleetBucket {
	type class struct {
		pw2, ph2 int
		members  []int
	}
	byKey := map[[2]int]*class{}
	var keys [][2]int
	for i, m := range ms {
		if m.varSum == 0 {
			continue
		}
		k := [2]int{nextPow2(m.w), nextPow2(m.h)}
		c := byKey[k]
		if c == nil {
			c = &class{pw2: k[0], ph2: k[1]}
			byKey[k] = c
			keys = append(keys, k)
		}
		c.members = append(c.members, i)
	}
	sort.Slice(keys, func(a, b int) bool {
		ka, kb := keys[a], keys[b]
		if aa, ab := ka[0]*ka[1], kb[0]*kb[1]; aa != ab {
			return aa > ab
		}
		if ka[0] != kb[0] {
			return ka[0] > kb[0]
		}
		return ka[1] > kb[1]
	})
	type bslot struct {
		pw2, ph2 int
		members  []int
	}
	var slots []*bslot
	for _, k := range keys {
		c := byKey[k]
		placed := false
		for _, s := range slots {
			if s.pw2 >= c.pw2 && s.ph2 >= c.ph2 {
				s.members = append(s.members, c.members...)
				placed = true
				break
			}
		}
		if !placed {
			slots = append(slots, &bslot{pw2: c.pw2, ph2: c.ph2, members: append([]int(nil), c.members...)})
		}
	}
	var out []fleetBucket
	for _, s := range slots {
		sort.Ints(s.members)
		b := fleetBucket{members: s.members}
		for _, mi := range s.members {
			b.tw = max(b.tw, ms[mi].w)
			b.th = max(b.th, ms[mi].h)
		}
		out = append(out, b)
	}
	return out
}

// FindAll searches parent for every member and returns their results in
// construction order. Panics like Find if any template is larger than
// parent.
func (f *Fleet) FindAll(parent image.Image) []Result {
	var pPix []uint8
	var pStride, pw, ph int
	var pOwned bool
	if f.step == 1 {
		pPix, pStride, pw, ph, pOwned = toGray(parent)
	} else {
		pPix, pStride, pw, ph, pOwned = toRGBA(parent)
	}
	defer func() {
		if pOwned {
			bytePool.put(pPix)
		}
	}()
	results := make([]Result, len(f.ms))
	for i, m := range f.ms {
		if m.w > pw || m.h > ph {
			panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d)", m.w, m.h, pw, ph))
		}
		if m.varSum == 0 { // flat members: constant answers, no transforms
			results[i] = Result{1, 0, 0, 1, 0, 0}
		}
	}
	if len(f.buckets) == 0 { // all-flat fleet
		return results
	}
	pConst := f.step == 4 && alphaConst(pPix, pStride, pw, ph)
	nthreads := threads()
	g := f.geom.Load()
	if g == nil || g.pw != pw || g.ph != ph || g.pConst != pConst {
		f.mu.Lock()
		if g = f.geom.Load(); g == nil || g.pw != pw || g.ph != ph || g.pConst != pConst {
			g = f.buildGeom(pw, ph, pConst, nthreads)
			f.geom.Store(g)
		}
		f.mu.Unlock()
	}

	for bi := range f.buckets {
		b, bg := &f.buckets[bi], &g.bg[bi]
		p := bg.p
		rwF, rhF := pw-b.tw+1, ph-b.th+1
		ntx := (rwF + p.blockW - 1) / p.blockW
		nty := (rhF + p.blockH - 1) / p.blockH
		ntiles := ntx * nty
		lastY0 := ((rhF - 1) / p.blockH) * p.blockH

		// Per-tile plan and frame-slab offsets: last-band tiles may run
		// the shrunk transform height (7.4-lite, decided once per bucket
		// from its th so every member shares the band geometry).
		specN := p.dftH * p.hw
		tileP := func(y0 int) *goPlan {
			if y0 == lastY0 && bg.p2 != nil {
				return bg.p2
			}
			return p
		}
		off := make([]int, ntiles+1)
		for t := 0; t < ntiles; t++ {
			bp := tileP(t / ntx * p.blockH)
			off[t+1] = off[t] + bg.cnMax*bp.dftH*bp.hw
		}
		frame := cplxPool.get(off[ntiles])

		// Phase 1: every tile's parent spectrum, once per bucket.
		team := 1
		if nw := min(nthreads, ntiles); teamEligible(p, bg.cnMax) {
			team = nthreads / nw
		}
		var nextT atomic.Int64
		runParallel(min(nthreads, ntiles), func(int) {
			z := cplxPool.get(p.dftW)
			defer cplxPool.put(z)
			for {
				t := int(nextT.Add(1) - 1)
				if t >= ntiles {
					return
				}
				x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
				bwF, bhF := min(p.blockW, rwF-x0), min(p.blockH, rhF-y0)
				bp := tileP(y0)
				sn := bp.dftH * bp.hw
				for k := 0; k < bg.cnMax; k++ {
					blockForwardGo(pPix[k:], pStride, f.step, x0, y0,
						bwF+b.tw-1, bhF+b.th-1, bp, frame[off[t]+k*sn:off[t]+(k+1)*sn], z, team)
				}
			}
		})

		// Phase 2: each member's tail — conjugate multiply against its
		// own spectrum (on a copy, the frame cache is read-only here),
		// inverse, emit into its own map, normalize, scan. Workers claim
		// members costliest-first from a shared queue (their outputs are
		// disjoint, so dynamic order cannot affect bits — codex finding,
		// PR #34); worker/team split derives from the bucket's member
		// count, and inverse teams respect the same eligibility gate as
		// the solo pipeline (codex findings, PR #34).
		outer := min(nthreads, len(b.members))
		inner := max(nthreads/outer, 1)
		var nextM atomic.Int64
		runParallel(outer, func(int) {
			spec := cplxPool.get(specN)
			z := cplxPool.get(p.dftW)
			defer cplxPool.put(spec)
			defer cplxPool.put(z)
			for {
				oi := int(nextM.Add(1) - 1)
				if oi >= len(bg.order) {
					return
				}
				pos := bg.order[oi]
				mi := b.members[pos]
				m := f.ms[mi]
				cn, set := g.cns[mi], bg.sets[pos]
				invTeam := 1
				if inner > 1 && teamEligible(p, cn) {
					invTeam = inner
				}
				rw, rh := pw-m.w+1, ph-m.h+1
				res := f32Pool.get(rw * rh)
				for t := 0; t < ntiles; t++ {
					x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
					bp, ts := p, set.spec
					if y0 == lastY0 && set.p2 != nil {
						bp, ts = set.p2, set.spec2
					}
					sn := bp.dftH * bp.hw
					// Interior tiles emit the shared block extent for
					// every member; the last tile per axis extends to the
					// member's own result edge — exactly the extra valid
					// width its smaller template earns from the shared
					// loads.
					bw, bh := p.blockW, p.blockH
					if t%ntx == ntx-1 {
						bw = rw - x0
					}
					if t/ntx == nty-1 {
						bh = rh - y0
					}
					for k := 0; k < cn; k++ {
						copy(spec[:sn], frame[off[t]+k*sn:off[t]+(k+1)*sn])
						mulConjGo(spec[:sn], ts[k*sn:(k+1)*sn])
						blockInverseEmitGo(bp, spec, z, res, rw, x0, y0, bw, bh, k > 0, invTeam)
					}
				}
				r := Result{}
				r.MinV, r.MinX, r.MinY, r.MaxV, r.MaxX, r.MaxY = normalizeParallelGo(
					pPix, pStride, pw, m.w, m.h, cn, f.step, rw, rh, inner,
					&m.tsum, m.varSum, res, res)
				results[mi] = r
				f32Pool.put(res)
			}
		})
		cplxPool.put(frame)
	}
	return results
}

// buildGeom plans every bucket's tile geometry for one parent size and
// builds every member's spectrum on its bucket's plan. Called under
// f.mu.
func (f *Fleet) buildGeom(pw, ph int, pConst bool, nthreads int) *fleetGeom {
	g := &fleetGeom{pw: pw, ph: ph, pConst: pConst,
		cns: make([]int, len(f.ms)), bg: make([]bucketGeom, len(f.buckets))}
	for i, m := range f.ms {
		if m.varSum == 0 {
			continue
		}
		cn := m.step
		if m.step == 4 {
			cn = 4
			if m.aConst && pConst {
				cn = 3
			}
		}
		g.cns[i] = cn
	}
	for bi := range f.buckets {
		b := &f.buckets[bi]
		bg := &g.bg[bi]
		rwF, rhF := pw-b.tw+1, ph-b.th+1
		p := newGoPlan(b.tw, b.th, rwF, rhF)
		bg.p = p
		if dh2 := shrinkDH(p, rhF, b.th); dh2 > 0 {
			q := *p
			q.dftH = dh2
			tabH := fftTables(dh2)
			q.triH, q.brevH = tabH.tri(), tabH.brev
			bg.p2 = &q
		}
		bg.sets = make([]*tspecSet, len(b.members))
		for pos, mi := range b.members {
			m := f.ms[mi]
			cn := g.cns[mi]
			bg.cnMax = max(bg.cnMax, cn)
			// shrinkTH = bucket th: every member's shrunk-band spectrum
			// lands on the one band geometry the shared forward uses.
			bg.sets[pos] = buildTSpecSet(m.pix, m.w*m.step, m.step, cn, m.w, m.h,
				rhF, b.th, nthreads, p, false)
		}
		// Costliest tails first, so the claim queue never strands one
		// expensive member on a worker while the rest idle out.
		bg.order = make([]int, len(b.members))
		for i := range bg.order {
			bg.order[i] = i
		}
		sort.SliceStable(bg.order, func(a, c int) bool {
			ma, mc := f.ms[b.members[bg.order[a]]], f.ms[b.members[bg.order[c]]]
			ca := int64(pw-ma.w+1) * int64(ph-ma.h+1) * int64(g.cns[b.members[bg.order[a]]])
			cc := int64(pw-mc.w+1) * int64(ph-mc.h+1) * int64(g.cns[b.members[bg.order[c]]])
			return ca > cc
		})
	}
	return g
}
