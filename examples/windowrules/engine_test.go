package main

import (
	"image"
	"strings"
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
		if !sameResult(first[i], again[i]) {
			t.Fatalf("rule %d drifted on re-run: %+v vs %+v", first[i].RuleID, first[i], again[i])
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

func sameResult(a, b RuleResult) bool {
	if a.Matched != b.Matched || (a.Best == nil) != (b.Best == nil) || len(a.All) != len(b.All) {
		return false
	}
	if a.Best != nil && *a.Best != *b.Best {
		return false
	}
	for i := range a.All {
		if a.All[i] != b.All[i] {
			return false
		}
	}
	return true
}

// The engine must keep private copies of the rule and template slices:
// a caller reloading its rule database after construction cannot
// desync the compiled groups (codex finding, PR #36 — the Fleet member
// slice contract, one level up).
func TestEngineInputIsolation(t *testing.T) {
	base := synthFrame(false)
	rules := demoRules(base)
	eng, err := NewEngine(rules)
	if err != nil {
		t.Fatal(err)
	}
	want, err := eng.Run(base)
	if err != nil {
		t.Fatal(err)
	}
	// The caller trashes everything it handed over.
	rules[0].Threshold = 2
	rules[0].Mode = ModeAll
	rules[0].Templates[1] = Template{ID: 999, Img: image.NewRGBA(image.Rect(0, 0, 8, 8))}
	rules[1].ROI = image.Rect(0, 0, 10, 10)
	rules[1].Templates = nil
	got, err := eng.Run(base)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if !sameResult(want[i], got[i]) {
			t.Fatalf("rule %d changed after caller mutation: %+v vs %+v", want[i].RuleID, want[i], got[i])
		}
	}
}

// Reversed ROI corners canonicalize to the intended region instead of
// reading as "empty = whole window" (codex finding, PR #36); a
// non-zero ROI with no area is a load error, never a silent
// whole-window search.
func TestEngineROIValidation(t *testing.T) {
	base := synthFrame(false)
	r := demoRules(base)[1] // tray-status, ROI (1150,0)-(1280,48)
	r.ROI = image.Rectangle{Min: image.Point{1280, 48}, Max: image.Point{1150, 0}}
	eng, err := NewEngine([]Rule{r})
	if err != nil {
		t.Fatal(err)
	}
	res, err := eng.Run(base)
	if err != nil {
		t.Fatal(err)
	}
	if !res[0].Matched || (res[0].Best.Rect.Min != wifiRect.Min && res[0].Best.Rect.Min != muteRect.Min) {
		t.Fatalf("reversed ROI: %+v", res[0])
	}
	r.ROI = image.Rect(10, 10, 10, 50) // zero width: no window can ever match inside it
	if _, err := NewEngine([]Rule{r}); err == nil {
		t.Fatal("degenerate ROI accepted")
	}
}

// Reusing one template ID at two different sizes fails the load: the
// deduped matcher keeps the first size, so honoring the second
// reference's size would desync fit from what FindAll actually runs —
// up to a Run-time panic (codex finding, PR #36 round 2). Same-size
// reuse across groups stays legal (the demo shares id 105 across two
// ROI groups).
func TestEngineTemplateSizeConflict(t *testing.T) {
	base := synthFrame(false)
	small := crop(base, wifiRect) // 24×24
	big := crop(base, panelRect)  // 300×200, same ID
	_, err := NewEngine([]Rule{
		{ID: 1, Mode: ModeFirst, Threshold: 0.9, ROI: image.Rect(0, 0, 100, 100),
			Templates: []Template{{ID: 5, Img: small}}},
		{ID: 2, Mode: ModeFirst, Threshold: 0.9,
			Templates: []Template{{ID: 5, Img: big}}},
	})
	if err == nil {
		t.Fatal("size-conflicting template id accepted")
	}
	if got := err.Error(); !strings.Contains(got, "template 5") || !strings.Contains(got, "300x200") {
		t.Fatalf("error lacks context: %q", got)
	}
}

// A template above the library's exact-statistics bound fails the load
// with rule/template context instead of panicking (codex finding,
// PR #36).
func TestEngineOversizedTemplateError(t *testing.T) {
	huge := image.NewRGBA(image.Rect(0, 0, 2442, 2442)) // 5,963,364 px > the cn=4 bound
	_, err := NewEngine([]Rule{{ID: 7, Mode: ModeFirst, Threshold: 0.9,
		Templates: []Template{{ID: 3, Img: huge}}}})
	if err == nil {
		t.Fatal("oversized template accepted")
	}
	if got := err.Error(); !strings.Contains(got, "rule 7") || !strings.Contains(got, "template 3") {
		t.Fatalf("error lacks context: %q", got)
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
