package cvmatch_test

import (
	"image"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

// fleetScene builds a screen-like parent with K exact crops of mixed
// sizes — the production shape (many templates, one frame).
func fleetScene(t testing.TB) (*image.RGBA, []*image.RGBA, []image.Point) {
	var desk *image.RGBA
	for _, s := range scenes.All("bench/testdata") {
		if s.Name == "window1600_button96x32" {
			desk = s.Parent
		}
	}
	if desk == nil {
		t.Fatal("desktop scene missing")
	}
	rng := rand.New(rand.NewSource(11))
	sizes := []image.Point{{24, 24}, {96, 32}, {48, 48}, {300, 200}, {16, 16}, {64, 128}}
	var subs []*image.RGBA
	var at []image.Point
	b := desk.Bounds()
	for i := 0; i < 12; i++ {
		sz := sizes[i%len(sizes)]
		x := rng.Intn(b.Dx() - sz.X)
		y := rng.Intn(b.Dy() - sz.Y)
		sub := image.NewRGBA(image.Rect(0, 0, sz.X, sz.Y))
		for r := 0; r < sz.Y; r++ {
			copy(sub.Pix[r*sub.Stride:r*sub.Stride+sz.X*4],
				desk.Pix[(y+r)*desk.Stride+x*4:(y+r)*desk.Stride+(x+sz.X)*4])
		}
		subs = append(subs, sub)
		at = append(at, image.Point{x, y})
	}
	return desk, subs, at
}

// FindAll must locate every crop exactly where solo Find does, with
// scores inside the plan-change tolerance class (the fleet's shared
// geometry may differ from each member's solo plan — same contract as
// the tile argmin itself).
func TestFleetMatchesFind(t *testing.T) {
	desk, subs, at := fleetScene(t)
	for _, gray := range []bool{false, true} {
		var ms []*cvmatch.Matcher
		for _, sub := range subs {
			if gray {
				ms = append(ms, cvmatch.NewGrayMatcher(sub))
			} else {
				ms = append(ms, cvmatch.NewMatcher(sub))
			}
		}
		fleet := cvmatch.NewFleet(ms...)
		got := fleet.FindAll(desk)
		hits := 0
		for i, r := range got {
			// Solo Find is the ground truth (a low-variance crop from a
			// flat desktop region can legitimately tie elsewhere first —
			// then both paths must agree on that same answer).
			_, _, _, sv, sx, sy := ms[i].Find(desk)
			if sx != r.MaxX || sy != r.MaxY {
				t.Fatalf("gray=%v member %d: fleet (%d,%d) vs solo (%d,%d)",
					gray, i, r.MaxX, r.MaxY, sx, sy)
			}
			if d := math.Abs(float64(r.MaxV - sv)); d > 1e-3 {
				t.Fatalf("gray=%v member %d: score drift %.2e vs solo", gray, i, d)
			}
			if r.MaxX == at[i].X && r.MaxY == at[i].Y {
				hits++
			}
		}
		if hits < len(got)*2/3 { // most crops are well-conditioned and land home
			t.Fatalf("gray=%v: only %d/%d crops found at their origin", gray, hits, len(got))
		}
	}
}

// A single-member fleet plans exactly like solo Find (same argmin
// inputs), so its result must be bit-identical — the anchor that pins
// the whole shared pipeline to the proven one.
func TestFleetSingleBitIdentical(t *testing.T) {
	for _, s := range scenes.All("bench/testdata") {
		if s.PX < 0 {
			continue
		}
		m := cvmatch.NewMatcher(s.Sub)
		want := tup(m.Find(s.Parent))
		r := cvmatch.NewFleet(m).FindAll(s.Parent)[0]
		got := tuple{r.MinV, r.MinX, r.MinY, r.MaxV, r.MaxX, r.MaxY}
		if got != want {
			t.Fatalf("%s: fleet %+v want %+v", s.Name, got, want)
		}
		g := cvmatch.NewGrayMatcher(s.Sub)
		wantG := tup(g.Find(s.Parent))
		rg := cvmatch.NewFleet(g).FindAll(s.Parent)[0]
		gotG := tuple{rg.MinV, rg.MinX, rg.MinY, rg.MaxV, rg.MaxX, rg.MaxY}
		if gotG != wantG {
			t.Fatalf("%s gray: fleet %+v want %+v", s.Name, gotG, wantG)
		}
	}
}

// The fleet's answers must not depend on thread count, member order,
// or repetition — bit-identical across all of it.
func TestFleetDeterminism(t *testing.T) {
	desk, subs, _ := fleetScene(t)
	var ms []*cvmatch.Matcher
	for _, sub := range subs {
		ms = append(ms, cvmatch.NewGrayMatcher(sub))
	}
	defer func(old int) { cvmatch.Threads = old }(cvmatch.Threads)
	cvmatch.Threads = 1
	base := cvmatch.NewFleet(ms...).FindAll(desk)
	for _, th := range []int{2, 4, 7} {
		cvmatch.Threads = th
		got := cvmatch.NewFleet(ms...).FindAll(desk)
		for i := range base {
			if got[i] != base[i] {
				t.Fatalf("threads=%d member %d: %+v vs %+v", th, i, got[i], base[i])
			}
		}
	}
	// Reversed member order, same per-member bits.
	cvmatch.Threads = 4
	rev := make([]*cvmatch.Matcher, len(ms))
	for i := range ms {
		rev[len(ms)-1-i] = ms[i]
	}
	gotRev := cvmatch.NewFleet(rev...).FindAll(desk)
	for i := range base {
		if gotRev[len(ms)-1-i] != base[i] {
			t.Fatalf("order: member %d moved bits", i)
		}
	}
	// Repeated FindAll on one fleet (cache hit path).
	fl := cvmatch.NewFleet(ms...)
	a := fl.FindAll(desk)
	b := fl.FindAll(desk)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("repeat: member %d differs", i)
		}
	}
}

// Concurrent FindAll calls (racing the lazy geometry build) must agree.
func TestFleetConcurrent(t *testing.T) {
	desk, subs, _ := fleetScene(t)
	var ms []*cvmatch.Matcher
	for _, sub := range subs[:6] {
		ms = append(ms, cvmatch.NewMatcher(sub))
	}
	fl := cvmatch.NewFleet(ms...)
	const workers = 4
	out := make([][]cvmatch.Result, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			out[w] = fl.FindAll(desk)
		}(w)
	}
	wg.Wait()
	for w := 1; w < workers; w++ {
		for i := range out[0] {
			if out[w][i] != out[0][i] {
				t.Fatalf("worker %d member %d: %+v vs %+v", w, i, out[w][i], out[0][i])
			}
		}
	}
}

// Mixed alpha: an alpha-varying member forces its own cn=4 while
// alpha-constant members keep cn=3 on a constant-alpha parent; every
// member must still land on its crop.
func TestFleetMixedAlpha(t *testing.T) {
	desk, subs, at := fleetScene(t)
	varAlpha := image.NewRGBA(subs[0].Bounds())
	copy(varAlpha.Pix, subs[0].Pix)
	for i := 3; i < len(varAlpha.Pix); i += 16 {
		varAlpha.Pix[i] = uint8(i) // template alpha varies; parent's is constant
	}
	ms := []*cvmatch.Matcher{cvmatch.NewMatcher(varAlpha), cvmatch.NewMatcher(subs[1]), cvmatch.NewMatcher(subs[2])}
	got := cvmatch.NewFleet(ms...).FindAll(desk)
	for i := 1; i < 3; i++ {
		if got[i].MaxX != at[i].X || got[i].MaxY != at[i].Y {
			t.Fatalf("member %d: (%d,%d) want %v", i, got[i].MaxX, got[i].MaxY, at[i])
		}
	}
	// The varying-alpha member runs cn=4 inside the fleet exactly as it
	// does solo — same location, tolerance-level score agreement.
	_, _, _, sv, sx, sy := ms[0].Find(desk)
	if got[0].MaxX != sx || got[0].MaxY != sy {
		t.Fatalf("varying-alpha member: fleet (%d,%d) vs solo (%d,%d)", got[0].MaxX, got[0].MaxY, sx, sy)
	}
	if d := math.Abs(float64(got[0].MaxV - sv)); d > 1e-3 {
		t.Fatalf("varying-alpha member: score drift %.2e vs solo", d)
	}
}

// Standing cost probe: the batch against per-member solo Finds.
func BenchmarkFleet(b *testing.B) {
	desk, subs, _ := fleetScene(b)
	var ms []*cvmatch.Matcher
	for _, sub := range subs {
		ms = append(ms, cvmatch.NewGrayMatcher(sub))
	}
	fl := cvmatch.NewFleet(ms...)
	fl.FindAll(desk)
	b.Run("FindAll12", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			fl.FindAll(desk)
		}
	})
	b.Run("SoloFinds12", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, m := range ms {
				m.Find(desk)
			}
		}
	})
}

// Flat members answer constant 1 without shaping the shared geometry
// (a large blank template must not inflate everyone's transforms), and
// an all-flat fleet returns without building anything.
func TestFleetFlatMembers(t *testing.T) {
	desk, subs, at := fleetScene(t)
	flat := image.NewRGBA(image.Rect(0, 0, 400, 300)) // larger than every crop
	for i := 0; i < len(flat.Pix); i += 4 {
		flat.Pix[i], flat.Pix[i+1], flat.Pix[i+2], flat.Pix[i+3] = 30, 30, 30, 255
	}
	ms := []*cvmatch.Matcher{cvmatch.NewMatcher(flat), cvmatch.NewMatcher(subs[0]), cvmatch.NewMatcher(subs[1])}
	got := cvmatch.NewFleet(ms...).FindAll(desk)
	if (got[0] != cvmatch.Result{1, 0, 0, 1, 0, 0}) {
		t.Fatalf("flat member: %+v", got[0])
	}
	for i := 1; i < 3; i++ {
		_, _, _, _, sx, sy := ms[i].Find(desk)
		if got[i].MaxX != sx || got[i].MaxY != sy {
			t.Fatalf("member %d: fleet (%d,%d) vs solo (%d,%d) [crop at %v]",
				i, got[i].MaxX, got[i].MaxY, sx, sy, at[i])
		}
	}
	all := cvmatch.NewFleet(cvmatch.NewMatcher(flat)).FindAll(desk)
	if (all[0] != cvmatch.Result{1, 0, 0, 1, 0, 0}) {
		t.Fatalf("all-flat fleet: %+v", all[0])
	}
}

// The fleet must keep a private copy of the member slice: mutating the
// caller's slice after construction cannot desync the cached geometry.
func TestFleetSliceIsolation(t *testing.T) {
	desk, subs, _ := fleetScene(t)
	ms := []*cvmatch.Matcher{cvmatch.NewGrayMatcher(subs[0]), cvmatch.NewGrayMatcher(subs[1])}
	fl := cvmatch.NewFleet(ms...)
	want := fl.FindAll(desk)
	ms[0] = cvmatch.NewGrayMatcher(subs[3]) // caller swaps a member afterwards
	got := fl.FindAll(desk)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("member %d changed after caller slice mutation", i)
		}
	}
}
