// Package bench compares cvmatch directly against native OpenCV C++
// (bench/cpp/native_bench, linked against prebuilt static OpenCV 4.12
// archives): element-wise response-map parity and per-scene maxVal checks.
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

// TestNativeValues cross-checks the peak that native OpenCV C++ and
// cvmatch report on every scene, including the low-score degraded/absent
// ones. Two gates: the peak VALUE must agree within 1e-4 (peak windows
// have the largest denominators, so their error is structurally the
// smallest in the map — measured worst 1.6e-6), and the peak LOCATION
// must be native's first-occurrence argmax, unless the near-tie clause
// fires: native's score at cvmatch's location is within 2e-3 of native's
// max (each side's map may independently wiggle the 1e-3 element gate, so
// two peaks closer than that can legitimately swap order).
func TestNativeValues(t *testing.T) {
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
			if v > cppMax { // strict: first occurrence, OpenCV's minMaxLoc order
				cppMax, cppI = v, i
			}
		}
		_, _, _, maxV, maxX, maxY := cvmatch.Match(s.Parent, s.Sub)
		d := math.Abs(float64(maxV - cppMax))
		loc := "ok"
		if maxX != cppI%ww || maxY != cppI/ww {
			nativeAtCv := math.Float32frombits(binary.LittleEndian.Uint32(data[12+(maxY*ww+maxX)*4:]))
			if gap := float64(cppMax) - float64(nativeAtCv); gap <= 2e-3 {
				loc = "tie"
			} else {
				loc = "MISMATCH"
				t.Errorf("%s: peak location (%d,%d) vs native (%d,%d) — native gap %.2e exceeds the tie clause",
					s.Name, maxX, maxY, cppI%ww, cppI/ww, gap)
			}
		}
		t.Logf("%-22s  C++ max=%.6f @(%d,%d)   cvmatch[%s] max=%.6f @(%d,%d)   |diff|=%.2e loc=%s",
			s.Name, cppMax, cppI%ww, cppI/ww, cvmatch.Impl, maxV, maxX, maxY, d, loc)
		if d > 1e-4 {
			t.Errorf("%s: maxVal diverges from native C++: %v vs %v", s.Name, maxV, cppMax)
		}
		checked++
	}
	if checked == 0 {
		t.Skip("no native result dumps present")
	}
}
