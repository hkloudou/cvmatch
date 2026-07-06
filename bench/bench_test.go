// Package bench compares cvmatch directly against native OpenCV C++
// (bench/cpp/native_bench, linked against prebuilt static OpenCV 4.12
// archives): element-wise response-map parity and three-way maxVal checks.
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

// TestThreeWayValues prints and cross-checks the maxVal that native OpenCV
// C++ and the active cvmatch core (cvmatch.Impl: "cgo" under the default
// build, "purego" under CGO_ENABLED=0 — run both to see all three) produce
// on every scene, including the low-score degraded/absent ones where the
// peak is far from 1.0. Values must agree within the float32 rounding
// envelope.
func TestThreeWayValues(t *testing.T) {
	checked := 0
	for _, s := range scenes.All("testdata") {
		data, err := os.ReadFile(filepath.Join("cpp", "scenes", s.Name+".result.raw"))
		if err != nil {
			continue
		}
		ww := int(int32(binary.LittleEndian.Uint32(data[4:])))
		cppMax, cppI := float32(-2), 0
		for i := 0; i < (len(data)-12)/4; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[12+i*4:]))
			if v > cppMax {
				cppMax, cppI = v, i
			}
		}
		_, _, _, maxV, maxX, maxY := cvmatch.Match(s.Parent, s.Sub)
		d := math.Abs(float64(maxV - cppMax))
		t.Logf("%-22s  C++ max=%.6f @(%d,%d)   cvmatch[%s] max=%.6f @(%d,%d)   |diff|=%.2e",
			s.Name, cppMax, cppI%ww, cppI/ww, cvmatch.Impl, maxV, maxX, maxY, d)
		if d > 1.5e-3 {
			t.Errorf("%s: maxVal diverges from native C++: %v vs %v", s.Name, maxV, cppMax)
		}
		checked++
	}
	if checked == 0 {
		t.Skip("no native result dumps present")
	}
}
