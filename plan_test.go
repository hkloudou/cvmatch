package cvmatch

import "testing"

// TestPlanPins pins the argmin planner's output for representative and
// degenerate shapes. The cost model is pure integer arithmetic, so these
// must reproduce on every platform and build — a differing plan would
// silently change output bits (goldens catch that too, but this test
// names the culprit). Update deliberately alongside a golden re-record
// when the cost model itself changes.
func TestPlanPins(t *testing.T) {
	cases := [][8]int{
		// tw, th, rw, rh -> dftW, dftH, blockW, blockH
		{128, 128, 1793, 953, 1024, 512, 897, 385},   // 1080p / 128 (noise1080p_sub128)
		{96, 32, 1505, 969, 512, 1024, 417, 969},     // window1600_button96x32
		{32, 32, 1889, 1049, 512, 256, 481, 225},     // noise1080p_sub32
		{256, 256, 3585, 1905, 2048, 1024, 1793, 769}, // noise4k_sub256
		{24, 24, 1577, 977, 256, 512, 233, 489},      // window1600_icon24x24
		{80, 80, 433, 401, 512, 512, 433, 401},       // photo_fruits (single tile)
		{64, 64, 449, 449, 512, 512, 449, 449},       // photo_baboon
		{100, 100, 769, 501, 512, 256, 413, 157},     // photo_building
		{128, 128, 625, 473, 512, 512, 385, 385},     // photo_starry_night
		{64, 64, 1, 1, 64, 64, 1, 1},                 // template == image, exact pow2 fit
		{1, 1, 50, 40, 4, 2, 4, 2},                   // 1x1 template
		{33, 257, 8, 44, 64, 512, 8, 44},             // tall template, tiny result width
		{17, 13, 81, 49, 64, 64, 48, 49},             // small odd template
		{1, 400, 2000, 1, 32, 512, 32, 1},            // 1-wide template, 1-row result
		{400, 1, 1, 2000, 512, 2, 1, 2},              // 1-tall template, 1-col result
	}
	for _, c := range cases {
		p := newGoPlan(c[0], c[1], c[2], c[3])
		if p.dftW != c[4] || p.dftH != c[5] || p.blockW != c[6] || p.blockH != c[7] {
			t.Errorf("plan(%dx%d in %dx%d result): got %dx%d blocks %dx%d, want %dx%d blocks %dx%d",
				c[0], c[1], c[2], c[3], p.dftW, p.dftH, p.blockW, p.blockH, c[4], c[5], c[6], c[7])
		}
	}
}
