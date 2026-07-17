// paritystat prints one machine-readable parity line per scene comparing
// cvmatch against the native OpenCV result dumps (produced by
// `cpp/native_bench cpp/scenes 1 dump`): the worst per-element |Δ|, the
// peak-value |Δ|, and the peak-location status (ok / tie / MISMATCH, same
// rules as bench's TestNativeValues). The bench-charts pipeline feeds
// these lines into the published matrix so drift toward the tolerance
// budget is visible on every refresh.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

func main() {
	dumps := flag.String("dumps", "cpp/scenes", "directory with *.result.raw native dumps")
	testdata := flag.String("testdata", "testdata", "scene image directory")
	flag.Parse()

	for _, s := range scenes.All(*testdata) {
		data, err := os.ReadFile(filepath.Join(*dumps, s.Name+".result.raw"))
		if err != nil {
			continue
		}
		ww := int(int32(binary.LittleEndian.Uint32(data[4:])))
		at := func(i int) float64 {
			return float64(math.Float32frombits(binary.LittleEndian.Uint32(data[12+i*4:])))
		}
		n := (len(data) - 12) / 4

		got, gw, _ := cvmatch.MatchMap(s.Parent, s.Sub)
		if gw != ww || len(got) != n {
			fmt.Printf("scene=%s error=dims\n", s.Name)
			continue
		}
		worst := 0.0
		nMax, nI := -2.0, 0
		for i := 0; i < n; i++ {
			v := at(i)
			if d := math.Abs(float64(got[i]) - v); d > worst {
				worst = d
			}
			if v > nMax { // strict: first occurrence, OpenCV's minMaxLoc order
				nMax, nI = v, i
			}
		}
		_, _, _, maxV, maxX, maxY := cvmatch.Match(s.Parent, s.Sub)
		loc := "ok"
		if maxX != nI%ww || maxY != nI/ww {
			if nMax-at(maxY*ww+maxX) <= 2e-3 {
				loc = "tie"
			} else {
				loc = "MISMATCH"
			}
		}
		fmt.Printf("scene=%s worst_abs=%.3e peak_dv=%.3e loc=%s\n",
			s.Name, worst, math.Abs(float64(maxV)-nMax), loc)
	}
}
