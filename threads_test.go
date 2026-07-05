package cvmatch

// Build-tag-free thread bit-identity tests: they run in BOTH builds, so the
// CGO_ENABLED=0 suite proves the pure-Go core's parallel decomposition on
// its own (impl_parity_test.go additionally cross-checks the C core when
// cgo is on).

import (
	"math/rand"
	"testing"
)

func TestThreadsBitIdenticalPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	iw, ih, tw, th := 700, 500, 96, 64 // several tiles, several bands
	img, tpl := randPix(iw*ih*4, rng), randPix(tw*th*4, rng)
	rw, rh := iw-tw+1, ih-th+1
	ref := make([]float32, rw*rh)
	rMinV, rMinX, rMinY, rMaxV, rMaxX, rMaxY := matchU8Go(img, iw*4, iw, ih, tpl, tw*4, tw, th, 3, 4, 1, ref)
	for _, n := range []int{2, 3, 4, 8} {
		got := make([]float32, rw*rh)
		minV, minX, minY, maxV, maxX, maxY := matchU8Go(img, iw*4, iw, ih, tpl, tw*4, tw, th, 3, 4, n, got)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("purego threads=%d: map differs at %d: %v vs %v", n, i, got[i], ref[i])
			}
		}
		if minV != rMinV || maxV != rMaxV || minX != rMinX || minY != rMinY || maxX != rMaxX || maxY != rMaxY {
			t.Fatalf("purego threads=%d: extrema differ", n)
		}
	}
}

func TestThreadsBitIdenticalExact(t *testing.T) {
	rng := rand.New(rand.NewSource(78))
	iw, ih, tw, th := 400, 300, 48, 40
	img, tpl := randPix(iw*ih*4, rng), randPix(tw*th*4, rng)
	rw, rh := iw-tw+1, ih-th+1
	ref := make([]float32, rw*rh)
	matchExactU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 4, 4, 1, ref)
	for _, n := range []int{2, 4} {
		got := make([]float32, rw*rh)
		matchExactU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 4, 4, n, got)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("exact threads=%d: map differs at %d", n, i)
			}
		}
	}
}
