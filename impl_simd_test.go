package cvmatch

import (
	"math"
	"math/rand"
	"testing"

	"github.com/hkloudou/cvmatch/internal/simd"
)

// TestSIMDMatchesScalar proves the assembly kernels are a pure speedup:
// with simd.Enabled toggled off, the generic Go loops must produce
// bit-identical response maps and extrema. (The kernels perform exactly
// the scalar single-rounded op sequence per lane, so any deviation is a
// bug.) On platforms without kernels the test degenerates to
// scalar==scalar and is skipped.
func TestSIMDMatchesScalar(t *testing.T) {
	if !simd.Enabled {
		t.Skip("no SIMD kernels on this platform")
	}
	defer func() { simd.Enabled = true }()

	rng := rand.New(rand.NewSource(123))
	cases := []struct{ iw, ih, tw, th, cn, step int }{
		{97, 61, 17, 13, 1, 1},
		{120, 90, 24, 18, 3, 4},
		{640, 400, 96, 32, 3, 4},
		{300, 200, 31, 29, 4, 4},
	}
	for ci, c := range cases {
		img := randPix(c.iw*c.ih*c.step, rng)
		tpl := randPix(c.tw*c.th*c.step, rng)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1

		simd.Enabled = true
		fast := make([]float32, rw*rh)
		fMinV, fMinX, fMinY, fMaxV, fMaxX, fMaxY := matchU8Go(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 3, fast)

		simd.Enabled = false
		slow := make([]float32, rw*rh)
		sMinV, sMinX, sMinY, sMaxV, sMaxX, sMaxY := matchU8Go(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 3, slow)

		for i := range fast {
			if math.Float32bits(fast[i]) != math.Float32bits(slow[i]) {
				t.Fatalf("case %d: SIMD map differs from scalar at %d: %v vs %v", ci, i, fast[i], slow[i])
			}
		}
		if fMinX != sMinX || fMinY != sMinY || fMaxX != sMaxX || fMaxY != sMaxY ||
			math.Float32bits(fMinV) != math.Float32bits(sMinV) ||
			math.Float32bits(fMaxV) != math.Float32bits(sMaxV) {
			t.Fatalf("case %d: SIMD extrema differ from scalar", ci)
		}
	}
}
