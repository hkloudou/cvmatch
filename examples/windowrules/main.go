// Command windowrules is the runnable reference for serving per-window
// match rules with cvmatch — the shape of a typical automation service:
// a windows table names each watched window, a matches table keyed by
// window_id lists its rules (template ids + ROI + threshold + mode),
// and an HTTP handler captures the window and reports every rule's
// verdict.
//
// Grafting it onto such a service is a 1:1 replacement of the per-call
// matching loop:
//
//	rule-DB load     → decode each blob once, build []Rule, one
//	                   NewEngine per window_id (rebuild on DB change)
//	capture          → any *image.RGBA screenshot of the window
//	per request      → results, err := engines[windowID].Run(frame)
//	JSON response    → RuleResult maps onto the usual per-rule payload;
//	                   Hit.Rect is window-relative, add the window's
//	                   screen origin for absolute coordinates
//
// The demo needs no capture stack or database: it synthesizes a
// deterministic 1280×800 "window", carves its templates from it, and
// runs three frames to show the cold build, the warm cache, and a
// widget moving. engine.go is the part to copy.
package main

import (
	"fmt"
	"image"
	"image/draw"
	"time"
)

// Widget geometry on the synthetic window (window coordinates).
var (
	okRect    = image.Rect(860, 700, 980, 736)
	okMoved   = image.Rect(600, 700, 720, 736)
	wifiRect  = image.Rect(1180, 12, 1204, 36)
	muteRect  = image.Rect(1216, 12, 1240, 36)
	panelRect = image.Rect(200, 200, 500, 400)
)

// synthFrame renders the fake window: a gradient background with a few
// bordered, textured widgets. moveButton relocates the confirm button —
// the "UI changed between frames" case.
func synthFrame(moveButton bool) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 1280, 800))
	for y := 0; y < 800; y++ {
		for x := 0; x < 1280; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8(30 + x*50/1280)
			img.Pix[i+1] = uint8(40 + y*50/800)
			img.Pix[i+2] = 64
			img.Pix[i+3] = 255
		}
	}
	drawWidget(img, wifiRect, 7)
	drawWidget(img, muteRect, 9)
	drawWidget(img, panelRect, 33)
	ok := okRect
	if moveButton {
		ok = okMoved
	}
	drawWidget(img, ok, 21)
	return img
}

// drawWidget fills r with a dark border and a deterministically
// textured interior (a plain LCG keeps the demo dependency-free and
// identical on every run — a stand-in for a real button or icon).
func drawWidget(img *image.RGBA, r image.Rectangle, seed uint32) {
	s := seed
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			s = s*1664525 + 1013904223
			i := img.PixOffset(x, y)
			if x == r.Min.X || y == r.Min.Y || x == r.Max.X-1 || y == r.Max.Y-1 {
				img.Pix[i+0], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 20, 20, 24, 255
				continue
			}
			img.Pix[i+0] = uint8(140 + (s>>8)&63)
			img.Pix[i+1] = uint8(120 + (s>>16)&63)
			img.Pix[i+2] = uint8(90 + (s>>24)&63)
			img.Pix[i+3] = 255
		}
	}
}

// crop copies a window region into a standalone template image — the
// demo's stand-in for a decoded PNG blob.
func crop(src *image.RGBA, r image.Rectangle) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), src, r.Min, draw.Src)
	return dst
}

// decoy renders a widget that exists nowhere on the frame — its best
// score stays far below any sane threshold.
func decoy(id int64, w, h int, seed uint32) Template {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	drawWidget(img, img.Bounds(), seed)
	return Template{ID: id, Img: img}
}

// demoRules builds the window's rule set from a rendered base frame,
// exercising every engine path: whole-window and ROI groups, color and
// gray, both modes, template reuse across rules, and an ROI too small
// for its template.
func demoRules(base *image.RGBA) []Rule {
	tpl := func(id int64, r image.Rectangle) Template {
		return Template{ID: id, Img: crop(base, r)}
	}
	panel := tpl(105, panelRect)
	return []Rule{
		// Whole window, ModeFirst: the first template is absent (fails),
		// the second is the confirm button (passes) — ordered semantics.
		{ID: 1, Name: "confirm-button", Mode: ModeFirst, Threshold: 0.95,
			Templates: []Template{decoy(100, 120, 36, 99), tpl(101, okRect)}},
		// Tray ROI, gray matching, ModeAll: two present icons compete,
		// the decoy is reported in All but never passes.
		{ID: 2, Name: "tray-status", ROI: image.Rect(1150, 0, 1280, 48),
			Gray: true, Mode: ModeAll, Threshold: 0.95,
			Templates: []Template{tpl(102, wifiRect), tpl(103, muteRect), decoy(104, 24, 24, 77)}},
		// Panel ROI, color: the search costs the ROI area, not the frame.
		{ID: 3, Name: "main-panel", ROI: image.Rect(180, 180, 520, 420),
			Mode: ModeFirst, Threshold: 0.95, Templates: []Template{panel}},
		// The same template id 105 in an ROI it cannot fit: one-shot
		// Match would panic; the engine reports "not matched".
		{ID: 4, Name: "oversized-roi", ROI: image.Rect(0, 0, 40, 40),
			Mode: ModeFirst, Threshold: 0.95, Templates: []Template{panel}},
	}
}

func main() {
	base := synthFrame(false)
	moved := synthFrame(true)
	eng, err := NewEngine(demoRules(base))
	if err != nil {
		panic(err)
	}
	frames := []struct {
		name  string
		frame *image.RGBA
	}{
		{"frame 1 (cold: plans + spectra build)", base},
		{"frame 2 (warm: every cache hits)", base},
		{"frame 3 (button moved)", moved},
	}
	for _, f := range frames {
		start := time.Now()
		results, err := eng.Run(f.frame)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s: %v\n", f.name, time.Since(start).Round(10*time.Microsecond))
		for _, rr := range results {
			if !rr.Matched {
				fmt.Printf("  rule %d %-14s ✗ not matched\n", rr.RuleID, rr.Name)
				continue
			}
			fmt.Printf("  rule %d %-14s ✓ tpl=%d score=%.4f at %v\n",
				rr.RuleID, rr.Name, rr.Best.TemplateID, rr.Best.Score, rr.Best.Rect)
		}
	}
}
