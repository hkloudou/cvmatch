// Package bench compares cvmatch against hkloudou/cv2 (bundled OpenCV).
// It lives in its own module so the main cvmatch module stays dependency-free.
package bench

import (
	"image"
	"math"
	"math/rand"
	"testing"

	cv2 "github.com/hkloudou/cv2"
	"github.com/hkloudou/cvmatch"
)

// makeImages builds a deterministic noisy-gradient parent and crops sub out
// of it at (px, py), so both libraries hunt for the same known target.
func makeImages(w, h, sw, sh, px, py int) (*image.RGBA, *image.RGBA) {
	rng := rand.New(rand.NewSource(1))
	parent := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := y*parent.Stride + x*4
			parent.Pix[o] = uint8((x*7 + rng.Intn(64)) & 0xff)
			parent.Pix[o+1] = uint8((y*5 + rng.Intn(64)) & 0xff)
			parent.Pix[o+2] = uint8((x*3 + y*2 + rng.Intn(64)) & 0xff)
			parent.Pix[o+3] = 255
		}
	}
	sub := image.NewRGBA(image.Rect(0, 0, sw, sh))
	for y := 0; y < sh; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+sw*4],
			parent.Pix[(py+y)*parent.Stride+px*4:])
	}
	return parent, sub
}

func TestParityWithCv2(t *testing.T) {
	parent, sub := makeImages(1280, 720, 96, 96, 431, 285)

	aMinV, _, _, aMaxV, aMaxX, aMaxY := cv2.Match(parent, sub)
	bMinV, _, _, bMaxV, bMaxX, bMaxY := cvmatch.Match(parent, sub)

	if aMaxX != bMaxX || aMaxY != bMaxY {
		t.Fatalf("maxLoc mismatch: cv2 (%d,%d) vs cvmatch (%d,%d)", aMaxX, aMaxY, bMaxX, bMaxY)
	}
	if d := math.Abs(float64(aMaxV - bMaxV)); d > 1e-3 {
		t.Fatalf("maxVal mismatch: cv2 %v vs cvmatch %v (diff %g)", aMaxV, bMaxV, d)
	}
	if d := math.Abs(float64(aMinV - bMinV)); d > 1e-3 {
		t.Fatalf("minVal mismatch: cv2 %v vs cvmatch %v (diff %g)", aMinV, bMinV, d)
	}

	_, _, _, gMaxV, gMaxX, gMaxY := cvmatch.MatchGray(parent, sub)
	if gMaxX != aMaxX || gMaxY != aMaxY || gMaxV < 0.999 {
		t.Fatalf("gray maxLoc mismatch: (%d,%d) %v", gMaxX, gMaxY, gMaxV)
	}
	t.Logf("cv2:     min=%.6f max=%.6f @(%d,%d)", aMinV, aMaxV, aMaxX, aMaxY)
	t.Logf("cvmatch: min=%.6f max=%.6f @(%d,%d)", bMinV, bMaxV, bMaxX, bMaxY)
	t.Logf("gray:    max=%.6f @(%d,%d)", gMaxV, gMaxX, gMaxY)
}

type scenario struct {
	name         string
	w, h, sw, sh int
	px, py       int
}

var scenarios = []scenario{
	{"720p_sub96", 1280, 720, 96, 96, 431, 285},
	{"1080p_sub128", 1920, 1080, 128, 128, 977, 604},
	{"1080p_sub32", 1920, 1080, 32, 32, 977, 604},
	{"4k_sub256", 3840, 2160, 256, 256, 1200, 900},
}

func benchAll(b *testing.B, fn func(parent, sub image.Image) (float32, int, int, float32, int, int)) {
	for _, s := range scenarios {
		parent, sub := makeImages(s.w, s.h, s.sw, s.sh, s.px, s.py)
		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _, _, maxV, maxX, maxY := fn(parent, sub)
				if maxX != s.px || maxY != s.py || maxV < 0.99 {
					b.Fatalf("bad match: (%d,%d) %v", maxX, maxY, maxV)
				}
			}
		})
	}
}

func BenchmarkCv2Match(b *testing.B)         { benchAll(b, cv2.Match) }
func BenchmarkCvmatchMatch(b *testing.B)     { benchAll(b, cvmatch.Match) }
func BenchmarkCvmatchMatchGray(b *testing.B) { benchAll(b, cvmatch.MatchGray) }
