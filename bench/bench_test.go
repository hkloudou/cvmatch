// Package bench compares cvmatch against hkloudou/cv2 (bundled OpenCV).
// It lives in its own module so the main cvmatch module stays dependency-free.
package bench

import (
	"encoding/binary"
	"image"
	"math"
	"math/rand"
	"testing"

	cv2 "github.com/hkloudou/cv2"
	"github.com/hkloudou/cvmatch"
)

// makeNoise builds a deterministic noisy-gradient parent and crops sub out
// of it at (px, py) — a worst case for matching: no flat areas, every pixel
// carries signal.
func makeNoise(w, h, sw, sh, px, py int) (*image.RGBA, *image.RGBA) {
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
	return parent, crop(parent, image.Rect(px, py, px+sw, py+sh))
}

type scenario struct {
	name    string
	parent  *image.RGBA
	sub     *image.RGBA
	px, py  int
	fullMap bool // include in the element-wise response-map parity test
}

func buildScenarios() []scenario {
	var list []scenario
	add := func(name string, parent, sub *image.RGBA, px, py int, fullMap bool) {
		list = append(list, scenario{name, parent, sub, px, py, fullMap})
	}

	// Realistic UI automation: find widgets inside a rendered desktop window
	// (title bar, toolbar, sidebar, text content, dialog with look-alike
	// buttons, large flat regions).
	ui := makeDesktop(1600, 1000)
	add("window1600_button96x32", ui.img, crop(ui.img, ui.button), ui.button.Min.X, ui.button.Min.Y, true)
	add("window1600_icon24x24", ui.img, crop(ui.img, ui.icon), ui.icon.Min.X, ui.icon.Min.Y, true)
	add("window1600_panel300x200", ui.img, crop(ui.img, ui.panel), ui.panel.Min.X, ui.panel.Min.Y, false)
	ui4k := makeDesktop(3840, 2160)
	add("window4k_button96x32", ui4k.img, crop(ui4k.img, ui4k.button), ui4k.button.Min.X, ui4k.button.Min.Y, false)

	// Dense noise (no flat regions), several image/template size ratios.
	p, s := makeNoise(1280, 720, 96, 96, 431, 285)
	add("noise720p_sub96", p, s, 431, 285, true)
	p, s = makeNoise(1920, 1080, 128, 128, 977, 604)
	add("noise1080p_sub128", p, s, 977, 604, false)
	p, s = makeNoise(1920, 1080, 32, 32, 977, 604)
	add("noise1080p_sub32", p, s, 977, 604, false)
	p, s = makeNoise(3840, 2160, 256, 256, 1200, 900)
	add("noise4k_sub256", p, s, 1200, 900, false)
	return list
}

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
	for _, s := range buildScenarios() {
		if !s.fullMap {
			continue
		}
		want, ww, wh := cv2FullMap(s.parent, s.sub)
		got, gw, gh := cvmatch.MatchMap(s.parent, s.sub)
		if ww != gw || wh != gh {
			t.Fatalf("%s: dims %dx%d vs %dx%d", s.name, ww, wh, gw, gh)
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
				s.name, worst, worstI%ww, worstI/ww, want[worstI], got[worstI])
		} else {
			t.Logf("%s: %dx%d map, worst element |diff|=%.2e", s.name, ww, wh, worst)
		}
	}
}

// TestParityWithCv2 checks the Match() min/max contract on every scenario.
func TestParityWithCv2(t *testing.T) {
	for _, s := range buildScenarios() {
		aMinV, _, _, aMaxV, aMaxX, aMaxY := cv2.Match(s.parent, s.sub)
		bMinV, _, _, bMaxV, bMaxX, bMaxY := cvmatch.Match(s.parent, s.sub)
		if aMaxX != bMaxX || aMaxY != bMaxY || aMaxX != s.px || aMaxY != s.py {
			t.Fatalf("%s: maxLoc cv2 (%d,%d) cvmatch (%d,%d) want (%d,%d)",
				s.name, aMaxX, aMaxY, bMaxX, bMaxY, s.px, s.py)
		}
		if d := math.Abs(float64(aMaxV - bMaxV)); d > 1e-3 {
			t.Fatalf("%s: maxVal cv2 %v cvmatch %v", s.name, aMaxV, bMaxV)
		}
		if d := math.Abs(float64(aMinV - bMinV)); d > 1e-3 {
			t.Fatalf("%s: minVal cv2 %v cvmatch %v", s.name, aMinV, bMinV)
		}
		_, _, _, gMaxV, gMaxX, gMaxY := cvmatch.MatchGray(s.parent, s.sub)
		if gMaxX != s.px || gMaxY != s.py || gMaxV < 0.999 {
			t.Fatalf("%s: gray maxLoc (%d,%d) %v", s.name, gMaxX, gMaxY, gMaxV)
		}
		t.Logf("%s: min=%.6f/%.6f max=%.6f/%.6f @(%d,%d)",
			s.name, aMinV, bMinV, aMaxV, bMaxV, aMaxX, aMaxY)
	}
}

func benchAll(b *testing.B, fn func(parent, sub image.Image) (float32, int, int, float32, int, int)) {
	for _, s := range buildScenarios() {
		b.Run(s.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _, _, maxV, maxX, maxY := fn(s.parent, s.sub)
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
