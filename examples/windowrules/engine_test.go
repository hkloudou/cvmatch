package main

import (
	"image"
	"testing"
)

// The demo scene doubles as the engine's regression net (CI runs it in
// both build modes): planted widgets must land exactly, both modes must
// keep their selection semantics, unevaluable templates must degrade to
// "not matched" instead of panicking, and repeated runs (the cache-hit
// path) must reproduce the verdicts.
func TestEngineDemoScene(t *testing.T) {
	base := synthFrame(false)
	eng, err := NewEngine(demoRules(base))
	if err != nil {
		t.Fatal(err)
	}
	first, err := eng.Run(base)
	if err != nil {
		t.Fatal(err)
	}

	// Rule 1 (whole window, ModeFirst): the decoy fails, the button wins.
	r := first[0]
	if !r.Matched || r.Best.TemplateID != 101 || r.Best.Rect.Min != okRect.Min {
		t.Fatalf("confirm-button: %+v", r)
	}
	// Rule 2 (tray ROI, gray, ModeAll): all three evaluated, an icon wins.
	r = first[1]
	if !r.Matched || len(r.All) != 3 {
		t.Fatalf("tray-status: %+v", r)
	}
	if r.Best.TemplateID != 102 && r.Best.TemplateID != 103 {
		t.Fatalf("tray-status best: %+v", r.Best)
	}
	want := map[int64]image.Point{102: wifiRect.Min, 103: muteRect.Min}
	for _, h := range r.All {
		if p, present := want[h.TemplateID]; present && h.Rect.Min != p {
			t.Fatalf("tray icon %d at %v want %v", h.TemplateID, h.Rect.Min, p)
		}
	}
	// Rule 3 (panel ROI): found at its planted spot, window coordinates.
	r = first[2]
	if !r.Matched || r.Best.Rect.Min != panelRect.Min {
		t.Fatalf("main-panel: %+v", r)
	}
	// Rule 4 (template larger than its ROI): unmatched, no panic.
	if first[3].Matched {
		t.Fatalf("oversized-roi matched: %+v", first[3])
	}

	// Warm re-run reproduces every verdict (fleet caches hit).
	again, err := eng.Run(base)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first {
		a, b := first[i], again[i]
		if a.Matched != b.Matched ||
			(a.Best == nil) != (b.Best == nil) ||
			(a.Best != nil && *a.Best != *b.Best) || len(a.All) != len(b.All) {
			t.Fatalf("rule %d drifted on re-run: %+v vs %+v", a.RuleID, a, b)
		}
	}

	// The moved frame relocates the button hit.
	moved, err := eng.Run(synthFrame(true))
	if err != nil {
		t.Fatal(err)
	}
	if r = moved[0]; !r.Matched || r.Best.Rect.Min != okMoved.Min {
		t.Fatalf("moved confirm-button: %+v", r)
	}
}

// A group whose members only partly fit the ROI runs the memoized
// sub-fleet with correct member↔rule mapping.
func TestEnginePartialFit(t *testing.T) {
	base := synthFrame(false)
	rules := []Rule{{
		ID: 10, Name: "partial", ROI: image.Rect(1150, 0, 1280, 48),
		Mode: ModeAll, Threshold: 0.95,
		Templates: []Template{
			{ID: 1, Img: crop(base, wifiRect)},  // 24×24: fits the 130×48 ROI
			{ID: 2, Img: crop(base, panelRect)}, // 300×200: cannot fit
		},
	}}
	eng, err := NewEngine(rules)
	if err != nil {
		t.Fatal(err)
	}
	for round := 0; round < 2; round++ { // second round hits the memoized sub-fleet
		res, err := eng.Run(base)
		if err != nil {
			t.Fatal(err)
		}
		r := res[0]
		if !r.Matched || len(r.All) != 1 || r.All[0].TemplateID != 1 ||
			r.Best.Rect.Min != wifiRect.Min {
			t.Fatalf("round %d: %+v", round, r)
		}
	}
}
