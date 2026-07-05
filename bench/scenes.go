package bench

import (
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
)

// Scene is one benchmark/parity case: find Sub inside Parent; the true
// location is (PX, PY).
type Scene struct {
	Name        string
	Parent, Sub *image.RGBA
	PX, PY      int
	FullMap     bool // include in the element-wise response-map parity test
}

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

// Scenarios returns every scene: synthetic UI screenshots, dense noise, and
// — when bench/testdata has been populated by fetch.sh — real photographs
// from OpenCV's samples/data.
func Scenarios() []Scene {
	var list []Scene
	add := func(name string, parent, sub *image.RGBA, px, py int, fullMap bool) {
		list = append(list, Scene{name, parent, sub, px, py, fullMap})
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

	return append(list, realScenes()...)
}

// realScenes loads photographs from OpenCV's samples/data (fetched on demand
// by bench/testdata/fetch.sh; scenes are skipped when the files are absent).
func realScenes() []Scene {
	_, thisFile, _, _ := runtime.Caller(0)
	dir := filepath.Join(filepath.Dir(thisFile), "testdata")
	cases := []struct {
		file           string
		sw, sh, px, py int
		fullMap        bool
	}{
		{"fruits.jpg", 80, 80, 305, 210, true},          // 512x480 fruit bowl
		{"baboon.jpg", 64, 64, 250, 180, true},          // 512x512 fur texture
		{"building.jpg", 100, 100, 420, 240, false},     // 868x600 architecture
		{"graf1.png", 120, 120, 350, 260, false},        // 800x640 graffiti wall
		{"starry_night.jpg", 128, 128, 400, 300, false}, // 1280x1001 painting
	}
	var list []Scene
	for _, c := range cases {
		f, err := os.Open(filepath.Join(dir, c.file))
		if err != nil {
			continue
		}
		src, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			continue
		}
		b := src.Bounds()
		parent := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(parent, parent.Bounds(), src, b.Min, draw.Src)
		name := "photo_" + c.file[:len(c.file)-len(filepath.Ext(c.file))]
		list = append(list, Scene{name, parent,
			crop(parent, image.Rect(c.px, c.py, c.px+c.sw, c.py+c.sh)),
			c.px, c.py, c.fullMap})
	}
	return list
}
