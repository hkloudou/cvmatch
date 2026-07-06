// Package bench compares cvmatch directly against native OpenCV C++
// (bench/cpp/native_bench, linked with the prebuilt static OpenCV that the
// hkloudou/cv2 module bundles). The Go-wrapper (cv2) comparison lived here
// through v1.2.x and settled its question — the wrapper adds only ~0-4%
// over native C++ — so the suite now measures against C++ directly.
package bench

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

// TestFullMapParityWithNativeCpp compares cvmatch.MatchMap element-by-element
// with response maps produced by the native C++ benchmark. Generate them with:
//
//	go run ./cmd/dumpscenes -out cpp/scenes
//	cpp/build.sh && cpp/native_bench cpp/scenes 1 dump
//
// The test skips scenes whose dump file is absent.
func TestFullMapParityWithNativeCpp(t *testing.T) {
	checked := 0
	for _, s := range scenes.All("testdata") {
		data, err := os.ReadFile(filepath.Join("cpp", "scenes", s.Name+".result.raw"))
		if err != nil {
			continue
		}
		if string(data[:4]) != "CVMR" {
			t.Fatalf("%s: bad result dump magic", s.Name)
		}
		ww := int(int32(binary.LittleEndian.Uint32(data[4:])))
		wh := int(int32(binary.LittleEndian.Uint32(data[8:])))
		got, gw, gh := cvmatch.MatchMap(s.Parent, s.Sub)
		if ww != gw || wh != gh {
			t.Fatalf("%s: dims %dx%d vs %dx%d", s.Name, ww, wh, gw, gh)
		}
		worst, worstI := 0.0, 0
		for i := range got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(data[12+i*4:]))
			d := math.Abs(float64(got[i]) - float64(want))
			if d > worst {
				worst, worstI = d, i
			}
		}
		if worst > 1e-3 {
			t.Errorf("%s: worst |diff|=%g at (%d,%d) vs native C++", s.Name, worst, worstI%ww, worstI/ww)
		} else {
			t.Logf("%s: %dx%d map vs native C++, worst |diff|=%.2e", s.Name, ww, wh, worst)
		}
		checked++
	}
	if checked == 0 {
		t.Skip("no native result dumps present (run cpp/native_bench ... dump)")
	}
}
