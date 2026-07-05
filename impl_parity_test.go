//go:build cgo

package cvmatch

import (
	"math"
	"math/rand"
	"testing"
)

// TestPureGoMatchesCgo compares the pure-Go core against the C core
// element-by-element (both response map and min/max contract) so the
// CGO_ENABLED=0 fallback provably computes the same thing.
func TestPureGoMatchesCgo(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	cases := []struct{ iw, ih, tw, th, cn, step int }{
		{97, 61, 17, 13, 1, 1},
		{97, 61, 17, 13, 4, 4},
		{120, 90, 24, 18, 3, 4},
		{300, 200, 31, 29, 1, 1},
		{64, 64, 64, 64, 1, 1},
		{50, 40, 1, 1, 4, 4},
		{640, 400, 96, 32, 3, 4}, // exercises multi-block tiling
	}
	for ci, c := range cases {
		img := randPix(c.iw*c.ih*c.step, rng)
		tpl := randPix(c.tw*c.th*c.step, rng)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1
		gotC := make([]float32, rw*rh)
		gotGo := make([]float32, rw*rh)
		cMinV, cMinX, cMinY, cMaxV, cMaxX, cMaxY := matchU8(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, gotC)
		gMinV, gMinX, gMinY, gMaxV, gMaxX, gMaxY := matchU8Go(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, gotGo)
		worst := 0.0
		for i := range gotC {
			if d := math.Abs(float64(gotC[i]) - float64(gotGo[i])); d > worst {
				worst = d
			}
		}
		if worst > 1e-5 {
			t.Fatalf("case %d: worst |cgo-purego| diff %g", ci, worst)
		}
		if cMinX != gMinX || cMinY != gMinY || cMaxX != gMaxX || cMaxY != gMaxY ||
			math.Abs(float64(cMinV-gMinV)) > 1e-5 || math.Abs(float64(cMaxV-gMaxV)) > 1e-5 {
			t.Fatalf("case %d: extrema mismatch cgo(%v@%d,%d %v@%d,%d) purego(%v@%d,%d %v@%d,%d)",
				ci, cMinV, cMinX, cMinY, cMaxV, cMaxX, cMaxY, gMinV, gMinX, gMinY, gMaxV, gMaxX, gMaxY)
		}
	}
}
