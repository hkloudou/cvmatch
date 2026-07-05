// Package bench compares cvmatch against hkloudou/cv2 (bundled OpenCV).
// It lives in its own module so the main cvmatch module stays dependency-free.
package bench

import (
	"encoding/binary"
	"image"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	cv2 "github.com/hkloudou/cv2"
	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

// cv2FullMap extracts OpenCV's complete CV_32F response map.
func cv2FullMap(parent, sub image.Image) ([]float32, int, int) {
	pm, err := cv2.ImageToMatRGBA(parent)
	if err != nil {
		panic(err)
	}
	defer pm.Close()
	sm, err := cv2.ImageToMatRGBA(sub)
	if err != nil {
		panic(err)
	}
	defer sm.Close()
	res := cv2.NewMat()
	defer res.Close()
	mask := cv2.NewMat()
	defer mask.Close()
	cv2.MatchTemplate(pm, sm, &res, cv2.TmCcoeffNormed, mask)
	data, err := res.ToBytes()
	if err != nil {
		panic(err)
	}
	w, h := res.Cols(), res.Rows()
	out := make([]float32, w*h)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, w, h
}

// TestFullMapParityWithCv2 compares every element of the response map
// against OpenCV's, on both UI and noise scenarios.
func TestFullMapParityWithCv2(t *testing.T) {
	for _, s := range scenes.All("testdata") {
		if !s.FullMap {
			continue
		}
		want, ww, wh := cv2FullMap(s.Parent, s.Sub)
		got, gw, gh := cvmatch.MatchMap(s.Parent, s.Sub)
		if ww != gw || wh != gh {
			t.Fatalf("%s: dims %dx%d vs %dx%d", s.Name, ww, wh, gw, gh)
		}
		worst, worstI := 0.0, 0
		for i := range want {
			d := math.Abs(float64(got[i]) - float64(want[i]))
			if d > worst {
				worst, worstI = d, i
			}
		}
		if worst > 1e-3 {
			t.Errorf("%s: worst |diff|=%g at (%d,%d): cv2=%v cvmatch=%v",
				s.Name, worst, worstI%ww, worstI/ww, want[worstI], got[worstI])
		} else {
			t.Logf("%s: %dx%d map, worst element |diff|=%.2e", s.Name, ww, wh, worst)
		}
	}
}

// TestParityWithCv2 checks the Match() min/max contract on every scenario.
func TestParityWithCv2(t *testing.T) {
	for _, s := range scenes.All("testdata") {
		aMinV, _, _, aMaxV, aMaxX, aMaxY := cv2.Match(s.Parent, s.Sub)
		bMinV, _, _, bMaxV, bMaxX, bMaxY := cvmatch.Match(s.Parent, s.Sub)
		if aMaxX != bMaxX || aMaxY != bMaxY || aMaxX != s.PX || aMaxY != s.PY {
			t.Fatalf("%s: maxLoc cv2 (%d,%d) cvmatch (%d,%d) want (%d,%d)",
				s.Name, aMaxX, aMaxY, bMaxX, bMaxY, s.PX, s.PY)
		}
		if d := math.Abs(float64(aMaxV - bMaxV)); d > 1e-3 {
			t.Fatalf("%s: maxVal cv2 %v cvmatch %v", s.Name, aMaxV, bMaxV)
		}
		if d := math.Abs(float64(aMinV - bMinV)); d > 1e-3 {
			t.Fatalf("%s: minVal cv2 %v cvmatch %v", s.Name, aMinV, bMinV)
		}
		_, _, _, gMaxV, gMaxX, gMaxY := cvmatch.MatchGray(s.Parent, s.Sub)
		if gMaxX != s.PX || gMaxY != s.PY || gMaxV < 0.999 {
			t.Fatalf("%s: gray maxLoc (%d,%d) %v", s.Name, gMaxX, gMaxY, gMaxV)
		}
		t.Logf("%s: min=%.6f/%.6f max=%.6f/%.6f @(%d,%d)",
			s.Name, aMinV, bMinV, aMaxV, bMaxV, aMaxX, aMaxY)
	}
}

func benchAll(b *testing.B, fn func(parent, sub image.Image) (float32, int, int, float32, int, int)) {
	for _, s := range scenes.All("testdata") {
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

func BenchmarkCv2Match(b *testing.B)         { benchAll(b, cv2.Match) }
func BenchmarkCvmatchMatch(b *testing.B)     { benchAll(b, cvmatch.Match) }
func BenchmarkCvmatchMatchGray(b *testing.B) { benchAll(b, cvmatch.MatchGray) }

// TestFullMapParityWithNativeCpp compares cvmatch.MatchMap element-by-element
// with response maps produced by the native C++ benchmark (same static
// OpenCV libs as cv2). Generate them with:
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

// cv2MatchStrips gives cv2/OpenCV the same multi-core opportunity cvmatch
// uses internally: the caller splits the parent into overlapping horizontal
// strips (overlap = template height - 1), matches them concurrently and
// merges the extrema in band order. This is the strongest fair 4-core
// baseline available to a cv2 user, since OpenCV's matchTemplate path
// cannot use threads itself.
func cv2MatchStrips(parent *image.RGBA, sub image.Image, nStrips int) (float32, int, int, float32, int, int) {
	pb, sb := parent.Bounds(), sub.Bounds()
	th := sb.Dy()
	rh := pb.Dy() - th + 1
	type ext struct {
		minV, maxV             float32
		minX, minY, maxX, maxY int
		ok                     bool
	}
	rs := make([]ext, nStrips)
	var wg sync.WaitGroup
	for i := 0; i < nStrips; i++ {
		y0, y1 := rh*i/nStrips, rh*(i+1)/nStrips
		if y1 <= y0 {
			continue
		}
		wg.Add(1)
		go func(i, y0, y1 int) {
			defer wg.Done()
			// cv2's API needs a tight buffer for a strip (its fast path
			// rejects sub-images), so the copy is part of the recipe's
			// honest cost. cvmatch needs no such copy: its internal
			// parallelism reads the original frame in place.
			w := pb.Dx()
			strip := image.NewRGBA(image.Rect(0, 0, w, y1-y0+th-1))
			for y := 0; y < y1-y0+th-1; y++ {
				copy(strip.Pix[y*strip.Stride:y*strip.Stride+w*4],
					parent.Pix[parent.PixOffset(pb.Min.X, pb.Min.Y+y0+y):])
			}
			minV, minX, minY, maxV, maxX, maxY := cv2.Match(strip, sub)
			rs[i] = ext{minV, maxV, minX, minY + y0, maxX, maxY + y0, true}
		}(i, y0, y1)
	}
	wg.Wait()
	var r ext
	for _, e := range rs { // band order preserves first-occurrence ties
		if !e.ok {
			continue
		}
		if !r.ok {
			r = e
			continue
		}
		if e.minV < r.minV {
			r.minV, r.minX, r.minY = e.minV, e.minX, e.minY
		}
		if e.maxV > r.maxV {
			r.maxV, r.maxX, r.maxY = e.maxV, e.maxX, e.maxY
		}
	}
	return r.minV, r.minX, r.minY, r.maxV, r.maxX, r.maxY
}

// BenchmarkCv2MatchStrips4 is cv2 with caller-side 4-way strip parallelism.
func BenchmarkCv2MatchStrips4(b *testing.B) {
	for _, s := range scenes.All("testdata") {
		b.Run(s.Name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _, _, maxV, maxX, maxY := cv2MatchStrips(s.Parent, s.Sub, 4)
				if maxX != s.PX || maxY != s.PY || maxV < 0.99 {
					b.Fatalf("bad match: (%d,%d) %v", maxX, maxY, maxV)
				}
			}
		})
	}
}
