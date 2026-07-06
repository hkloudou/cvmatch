package cvmatch

// Build-tag-free thread bit-identity tests: they run in BOTH builds, so the
// CGO_ENABLED=0 suite proves the pure-Go core's parallel decomposition on
// its own (impl_parity_test.go additionally cross-checks the C core when
// cgo is on).

import (
	"image"
	"math/rand"
	"runtime"
	"testing"
)

// TestThreadsVar covers the public Threads switch: clamping, and that any
// setting reaches the public API with bit-identical results.
func TestThreadsVar(t *testing.T) {
	defer func(old int) { Threads = old }(Threads)

	Threads = 0
	want := runtime.GOMAXPROCS(0)
	if want > 16 {
		want = 16
	}
	if got := threads(); got != want {
		t.Fatalf("Threads=0: threads()=%d, want GOMAXPROCS-capped %d", got, want)
	}
	Threads = 99
	if got := threads(); got != 16 {
		t.Fatalf("Threads=99: threads()=%d, want clamp to 16", got)
	}
	Threads = -3
	if got := threads(); got != want {
		t.Fatalf("Threads=-3: threads()=%d, want automatic %d", got, want)
	}

	rng := rand.New(rand.NewSource(88))
	parent := image.NewRGBA(image.Rect(0, 0, 300, 220))
	copy(parent.Pix, randPix(len(parent.Pix), rng))
	sub := image.NewRGBA(image.Rect(0, 0, 48, 36))
	copy(sub.Pix, randPix(len(sub.Pix), rng))

	Threads = 1
	ref, w, h := MatchMap(parent, sub)
	rMinV, rMinX, rMinY, rMaxV, rMaxX, rMaxY := Match(parent, sub)
	for _, n := range []int{2, 5, 16} {
		Threads = n
		got, gw, gh := MatchMap(parent, sub)
		if gw != w || gh != h {
			t.Fatalf("Threads=%d: map size %dx%d, want %dx%d", n, gw, gh, w, h)
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("Threads=%d: map differs at %d: %v vs %v", n, i, got[i], ref[i])
			}
		}
		minV, minX, minY, maxV, maxX, maxY := Match(parent, sub)
		if minV != rMinV || maxV != rMaxV || minX != rMinX || minY != rMinY || maxX != rMaxX || maxY != rMaxY {
			t.Fatalf("Threads=%d: Match extrema differ from Threads=1", n)
		}
	}
}

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
