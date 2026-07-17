//go:build cgo

package cvmatch

import (
	"math"
	"math/rand"
	"testing"
)

// TestPureGoMatchesCgo compares the pure-Go core against the C core
// element-by-element. Both cores execute the same single-rounded IEEE op
// sequence (shared deterministic twiddles, pinned fp-contract, explicit
// float32 roundings in Go), so the response maps and extrema must be
// BIT-IDENTICAL, not merely close.
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
		cMinV, cMinX, cMinY, cMaxV, cMaxX, cMaxY := matchU8(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 4, gotC)
		gMinV, gMinX, gMinY, gMaxV, gMaxX, gMaxY := matchU8Go(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 4, gotGo)
		for i := range gotC {
			if math.Float32bits(gotC[i]) != math.Float32bits(gotGo[i]) {
				t.Fatalf("case %d: map differs at %d: cgo %v (%08x) vs purego %v (%08x)",
					ci, i, gotC[i], math.Float32bits(gotC[i]), gotGo[i], math.Float32bits(gotGo[i]))
			}
		}
		if cMinX != gMinX || cMinY != gMinY || cMaxX != gMaxX || cMaxY != gMaxY ||
			math.Float32bits(cMinV) != math.Float32bits(gMinV) ||
			math.Float32bits(cMaxV) != math.Float32bits(gMaxV) {
			t.Fatalf("case %d: extrema mismatch cgo(%v@%d,%d %v@%d,%d) purego(%v@%d,%d %v@%d,%d)",
				ci, cMinV, cMinX, cMinY, cMaxV, cMaxX, cMaxY, gMinV, gMinX, gMinY, gMaxV, gMaxX, gMaxY)
		}
	}
}

// TestThreadsBitIdentical proves the parallel decomposition changes nothing:
// any worker count produces byte-identical response maps and extrema in both
// cores.
func TestThreadsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	iw, ih, tw, th := 700, 500, 96, 64 // several tiles, several bands
	img, tpl := randPix(iw*ih*4, rng), randPix(tw*th*4, rng)
	rw, rh := iw-tw+1, ih-th+1
	type core struct {
		name string
		fn   func(threads int, res []float32) (float32, int, int, float32, int, int)
	}
	cores := []core{
		{"cgo", func(n int, res []float32) (float32, int, int, float32, int, int) {
			return matchU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 3, 4, n, res)
		}},
	}
	for _, c := range cores {
		ref := make([]float32, rw*rh)
		rMinV, rMinX, rMinY, rMaxV, rMaxX, rMaxY := c.fn(1, ref)
		for _, n := range []int{2, 3, 4, 8} {
			got := make([]float32, rw*rh)
			minV, minX, minY, maxV, maxX, maxY := c.fn(n, got)
			for i := range ref {
				if got[i] != ref[i] {
					t.Fatalf("%s threads=%d: map differs at %d: %v vs %v", c.name, n, i, got[i], ref[i])
				}
			}
			if minV != rMinV || maxV != rMaxV || minX != rMinX || minY != rMinY || maxX != rMaxX || maxY != rMaxY {
				t.Fatalf("%s threads=%d: extrema differ", c.name, n)
			}
		}
	}
}
