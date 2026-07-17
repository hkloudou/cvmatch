package cvmatch_test

import (
	"image"
	"testing"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

// Benchmarks over the shared scene set (byte-identical to what the native
// OpenCV C++ benchmark in bench/ consumes):
//
//	go test -bench . -benchtime 5x
//
// The photo scenes require bench/testdata/fetch.sh to have run.
func benchScenes(b *testing.B, fn func(parent, sub image.Image) (float32, int, int, float32, int, int)) {
	for _, s := range scenes.All("bench/testdata") {
		if s.PX < 0 { // low-score parity scenes have no position to assert
			continue
		}
		b.Run(s.Name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _, _, maxV, maxX, maxY := fn(s.Parent, s.Sub)
				if maxX != s.PX || maxY != s.PY || maxV < 0.99 {
					b.Fatalf("bad match: (%d,%d) %v", maxX, maxY, maxV)
				}
			}
		})
	}
}

func BenchmarkMatch(b *testing.B)     { benchScenes(b, cvmatch.Match) }
func BenchmarkMatchGray(b *testing.B) { benchScenes(b, cvmatch.MatchGray) }
