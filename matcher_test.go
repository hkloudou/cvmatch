package cvmatch_test

import (
	"image"
	"sync"
	"testing"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

type tuple struct {
	minV       float32
	minX, minY int
	maxV       float32
	maxX, maxY int
}

func tup(a float32, b, c int, d float32, e, f int) tuple { return tuple{a, b, c, d, e, f} }

// The Matcher API must reproduce the one-shot entrypoints bit for bit:
// same tuple on every scene, stable across repeated Finds (cache hits),
// across geometry changes (cache rebuilds), and under concurrent use.
func TestMatcherMatchesMatch(t *testing.T) {
	for _, s := range scenes.All("bench/testdata") {
		want := tup(cvmatch.Match(s.Parent, s.Sub))
		m := cvmatch.NewMatcher(s.Sub)
		for i := 0; i < 3; i++ {
			if got := tup(m.Find(s.Parent)); got != want {
				t.Fatalf("%s color find %d: got %+v want %+v", s.Name, i, got, want)
			}
		}
		wantG := tup(cvmatch.MatchGray(s.Parent, s.Sub))
		g := cvmatch.NewGrayMatcher(s.Sub)
		for i := 0; i < 3; i++ {
			if got := tup(g.Find(s.Parent)); got != wantG {
				t.Fatalf("%s gray find %d: got %+v want %+v", s.Name, i, got, wantG)
			}
		}
	}
}

// Alternating parent geometries forces cache rebuilds each call; every
// answer must still equal the one-shot result for that exact parent.
func TestMatcherGeometryChange(t *testing.T) {
	all := scenes.All("bench/testdata")
	var s scenes.Scene
	for _, c := range all {
		if c.Name == "noise720p_sub96" {
			s = c
			break
		}
	}
	full := s.Parent
	crop := full.SubImage(image.Rect(0, 0, full.Bounds().Dx()-40, full.Bounds().Dy()-24)).(*image.RGBA)
	m := cvmatch.NewMatcher(s.Sub)
	for i := 0; i < 2; i++ {
		for _, parent := range []image.Image{full, crop} {
			want := tup(cvmatch.Match(parent, s.Sub))
			if got := tup(m.Find(parent)); got != want {
				t.Fatalf("geometry %v round %d: got %+v want %+v", parent.Bounds(), i, got, want)
			}
		}
	}
}

// Concurrent Finds on one Matcher (first call races the lazy build)
// must all return the identical tuple. Run under -race in CI.
func TestMatcherConcurrent(t *testing.T) {
	all := scenes.All("bench/testdata")
	var s scenes.Scene
	for _, c := range all {
		if c.Name == "noise640_alpha" {
			s = c
			break
		}
	}
	want := tup(cvmatch.Match(s.Parent, s.Sub))
	m := cvmatch.NewMatcher(s.Sub)
	const workers = 8
	got := make([]tuple, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			got[w] = tup(m.Find(s.Parent))
		}(w)
	}
	wg.Wait()
	for w := 0; w < workers; w++ {
		if got[w] != want {
			t.Fatalf("worker %d: got %+v want %+v", w, got[w], want)
		}
	}
}

// A perfectly flat template scores 1 everywhere through both paths.
func TestMatcherFlatTemplate(t *testing.T) {
	parent := image.NewRGBA(image.Rect(0, 0, 64, 48))
	for i := range parent.Pix {
		parent.Pix[i] = uint8(i * 7)
	}
	flat := image.NewRGBA(image.Rect(0, 0, 8, 8))
	for i := 0; i < len(flat.Pix); i += 4 {
		flat.Pix[i], flat.Pix[i+1], flat.Pix[i+2], flat.Pix[i+3] = 9, 9, 9, 255
	}
	want := tup(cvmatch.Match(parent, flat))
	if got := tup(cvmatch.NewMatcher(flat).Find(parent)); got != want {
		t.Fatalf("flat: got %+v want %+v", got, want)
	}
}

// Standing cost probe: Find vs Match on a prep-heavy scene (the panel
// template is large relative to its plan tile, so the cached spectrum
// is worth ~10% — see the Phase 11 ledger for the per-shape table).
func BenchmarkFind(b *testing.B) {
	for _, name := range []string{"window1600_panel300x200", "photo_fruits"} {
		for _, s := range scenes.All("bench/testdata") {
			if s.Name != name {
				continue
			}
			m := cvmatch.NewMatcher(s.Sub)
			m.Find(s.Parent)
			b.Run(name, func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					m.Find(s.Parent)
				}
			})
		}
	}
}
