package cvmatch

import (
	"math/rand"
	"sync"
	"testing"
)

// TestConcurrentMatch pins down the thread-safety contract: every public
// function is safe for concurrent use (no shared mutable state, per-call
// scratch), so callers get throughput scaling by simply running Match calls
// in goroutines. Run with -race.
func TestConcurrentMatch(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	parent, sub := makeParentSub(320, 240, 32, 24, 100, 80, rng)
	_, _, _, wantV, wantX, wantY := Match(parent, sub)
	_, _, _, gWantV, gWantX, gWantY := MatchGray(parent, sub)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				_, _, _, v, x, y := Match(parent, sub)
				if x != wantX || y != wantY || v != wantV {
					t.Errorf("concurrent Match diverged: (%d,%d) %v", x, y, v)
				}
				_, _, _, gv, gx, gy := MatchGray(parent, sub)
				if gx != gWantX || gy != gWantY || gv != gWantV {
					t.Errorf("concurrent MatchGray diverged: (%d,%d) %v", gx, gy, gv)
				}
				res, w, _ := MatchMap(parent, sub)
				if res[wantY*w+wantX] != wantV {
					t.Errorf("concurrent MatchMap diverged at (%d,%d)", wantX, wantY)
				}
			}
		}()
	}
	wg.Wait()
}
