// Package scenes builds the deterministic benchmark/parity scenes shared by
// the native-C++ comparison module (bench/) and the main module's
// benchmarks: synthetic desktop screenshots, dense noise, and real
// photographs from OpenCV samples/data when present.
package scenes

import (
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"
	"path/filepath"
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

// degrade returns a copy of tpl with deterministic per-byte RGB noise in
// [-amp, +amp], simulating a stale/compressed reference image whose match
// score should land well below 1.0.
func degrade(tpl *image.RGBA, amp int) *image.RGBA {
	rng := rand.New(rand.NewSource(5))
	out := image.NewRGBA(tpl.Bounds())
	copy(out.Pix, tpl.Pix)
	for i := 0; i < len(out.Pix); i += 4 {
		for k := 0; k < 3; k++ {
			v := int(out.Pix[i+k]) + rng.Intn(2*amp+1) - amp
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			out.Pix[i+k] = uint8(v)
		}
	}
	return out
}

// makeNoiseAlpha is makeNoise with a non-constant alpha plane.
func makeNoiseAlpha(w, h, sw, sh, px, py int) (*image.RGBA, *image.RGBA) {
	rng := rand.New(rand.NewSource(2))
	parent := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(parent.Pix); i += 4 {
		parent.Pix[i] = uint8(rng.Intn(256))
		parent.Pix[i+1] = uint8(rng.Intn(256))
		parent.Pix[i+2] = uint8(rng.Intn(256))
		parent.Pix[i+3] = uint8(64 + rng.Intn(192))
	}
	return parent, crop(parent, image.Rect(px, py, px+sw, py+sh))
}

// Scenarios returns every scene: synthetic UI screenshots, dense noise, and
// — when bench/testdata has been populated by fetch.sh — real photographs
// from OpenCV's samples/data.
func All(testdata string) []Scene {
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
	p720, s720 := makeNoise(1280, 720, 96, 96, 431, 285)
	add("noise720p_sub96", p720, s720, 431, 285, true)
	p, s := makeNoise(1920, 1080, 128, 128, 977, 604)
	add("noise1080p_sub128", p, s, 977, 604, false)
	p, s = makeNoise(1920, 1080, 32, 32, 977, 604)
	add("noise1080p_sub32", p, s, 977, 604, false)
	p, s = makeNoise(3840, 2160, 256, 256, 1200, 900)
	add("noise4k_sub256", p, s, 1200, 900, false)

	// Varying alpha: defeats the constant-channel skip so the full
	// 4-channel path runs, and pins it element-wise against real OpenCV
	// (which treats alpha as a data channel on CV_8UC4 input).
	p, s = makeNoiseAlpha(640, 480, 64, 48, 217, 143)
	add("noise640_alpha", p, s, 217, 143, true)

	// Low-score scenes (PX = -1: no position to assert, excluded from the
	// speed benchmarks). Every other scene peaks at ~1.0, which leaves the
	// mid-range of the response map unpinned; "degraded" perturbs a real
	// crop so the peak lands well below 1, "half_degraded" pushes it to
	// mid-range, and "absent" searches a patch that is not in the image at
	// all, so the whole map is noise-level. All carry FullMap so native-C++
	// parity covers those value ranges.
	add("degraded_button", ui.img, degrade(crop(ui.img, ui.button), 40), -1, -1, true)
	add("half_degraded_noise", p720, degrade(s720, 170), -1, -1, true)
	np, _ := makeNoise(256, 256, 1, 1, 0, 0)
	add("absent_patch", ui.img, crop(np, image.Rect(60, 60, 140, 120)), -1, -1, true)

	return append(list, realScenes(testdata)...)
}

// realScenes loads photographs from OpenCV's samples/data (fetched on demand
// by bench/testdata/fetch.sh; scenes are skipped when the files are absent).
func realScenes(dir string) []Scene {
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

// This file synthesizes a deterministic desktop-style screenshot so the
// benchmarks cover the workload cvmatch is built for: hunting a button, a
// toolbar icon or a dialog inside a real window, with large flat regions,
// gradients, text-like content and repeated-but-not-identical widgets.

type uiTargets struct {
	img    *image.RGBA
	button image.Rectangle // an "OK"-style dialog button
	icon   image.Rectangle // a small toolbar icon
	panel  image.Rectangle // a large content region
}

func fillRect(img *image.RGBA, r image.Rectangle, c color.RGBA) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		o := img.PixOffset(r.Min.X, y)
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Pix[o] = c.R
			img.Pix[o+1] = c.G
			img.Pix[o+2] = c.B
			img.Pix[o+3] = 255
			o += 4
		}
	}
}

func vGradient(img *image.RGBA, r image.Rectangle, top, bottom color.RGBA) {
	h := r.Dy() - 1
	if h < 1 {
		h = 1
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		t := float64(y-r.Min.Y) / float64(h)
		c := color.RGBA{
			uint8(float64(top.R) + t*float64(int(bottom.R)-int(top.R))),
			uint8(float64(top.G) + t*float64(int(bottom.G)-int(top.G))),
			uint8(float64(top.B) + t*float64(int(bottom.B)-int(top.B))),
			255,
		}
		fillRect(img, image.Rect(r.Min.X, y, r.Max.X, y+1), c)
	}
}

func outline(img *image.RGBA, r image.Rectangle, c color.RGBA) {
	fillRect(img, image.Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+1), c)
	fillRect(img, image.Rect(r.Min.X, r.Max.Y-1, r.Max.X, r.Max.Y), c)
	fillRect(img, image.Rect(r.Min.X, r.Min.Y, r.Min.X+1, r.Max.Y), c)
	fillRect(img, image.Rect(r.Max.X-1, r.Min.Y, r.Max.X, r.Max.Y), c)
}

// textNoise draws random dark pixel runs that mimic rendered text lines.
func textNoise(img *image.RGBA, r image.Rectangle, lineH int, rng *rand.Rand) {
	for y := r.Min.Y + 4; y+lineH < r.Max.Y; y += lineH + 6 {
		x := r.Min.X + 8
		for x < r.Max.X-20 {
			run := 3 + rng.Intn(24)
			if x+run > r.Max.X-8 {
				run = r.Max.X - 8 - x
			}
			shade := uint8(30 + rng.Intn(90))
			for yy := 0; yy < lineH-4; yy++ {
				for xx := 0; xx < run; xx++ {
					if rng.Intn(3) != 0 {
						o := img.PixOffset(x+xx, y+yy)
						img.Pix[o], img.Pix[o+1], img.Pix[o+2] = shade, shade, shade
					}
				}
			}
			x += run + 4 + rng.Intn(14)
		}
	}
}

// drawButton renders a bordered gradient button whose "label" pixels come
// from seed, so multiple buttons look alike without being identical (real
// windows are full of near-duplicate widgets; matching must still pick the
// exact one).
func drawButton(img *image.RGBA, r image.Rectangle, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	outline(img, r, color.RGBA{118, 118, 118, 255})
	inner := image.Rect(r.Min.X+1, r.Min.Y+1, r.Max.X-1, r.Max.Y-1)
	vGradient(img, inner, color.RGBA{243, 243, 243, 255}, color.RGBA{214, 214, 214, 255})
	label := image.Rect(r.Min.X+12, r.Min.Y+r.Dy()/2-5, r.Max.X-12, r.Min.Y+r.Dy()/2+5)
	textNoise(img, image.Rect(label.Min.X, label.Min.Y-4, label.Max.X, label.Max.Y+6), 10, rng)
}

func drawIcon(img *image.RGBA, r image.Rectangle, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	for y := r.Min.Y; y < r.Max.Y; y += 4 {
		for x := r.Min.X; x < r.Max.X; x += 4 {
			c := color.RGBA{uint8(rng.Intn(256)), uint8(rng.Intn(256)), uint8(rng.Intn(256)), 255}
			fillRect(img, image.Rect(x, y, min(x+4, r.Max.X), min(y+4, r.Max.Y)), c)
		}
	}
	outline(img, r, color.RGBA{90, 90, 90, 255})
}

// makeDesktop renders a w x h screenshot: desktop background, a main window
// with title bar, toolbar icons, sidebar, text content, and a dialog with
// three buttons. Returns the crop rectangles used as templates.
func makeDesktop(w, h int) uiTargets {
	rng := rand.New(rand.NewSource(3))
	img := image.NewRGBA(image.Rect(0, 0, w, h))

	fillRect(img, img.Bounds(), color.RGBA{0, 120, 160, 255}) // desktop
	vGradient(img, image.Rect(0, h-48, w, h), color.RGBA{40, 40, 48, 255}, color.RGBA{24, 24, 30, 255})

	win := image.Rect(w/16, h/20, w-w/16, h-h/10)
	fillRect(img, win, color.RGBA{255, 255, 255, 255})
	outline(img, win, color.RGBA{100, 100, 100, 255})
	title := image.Rect(win.Min.X, win.Min.Y, win.Max.X, win.Min.Y+34)
	vGradient(img, title, color.RGBA{70, 130, 200, 255}, color.RGBA{50, 100, 170, 255})

	// toolbar with a row of distinct icons
	toolbar := image.Rect(win.Min.X, title.Max.Y, win.Max.X, title.Max.Y+36)
	vGradient(img, toolbar, color.RGBA{240, 240, 240, 255}, color.RGBA{222, 222, 222, 255})
	var icon image.Rectangle
	for i := 0; i < 18; i++ {
		r := image.Rect(win.Min.X+10+i*34, toolbar.Min.Y+6, win.Min.X+10+i*34+24, toolbar.Min.Y+30)
		drawIcon(img, r, int64(100+i))
		if i == 11 {
			icon = r
		}
	}

	// sidebar with items
	side := image.Rect(win.Min.X, toolbar.Max.Y, win.Min.X+win.Dx()/5, win.Max.Y)
	fillRect(img, side, color.RGBA{245, 245, 247, 255})
	for i := 0; i < 14; i++ {
		y := side.Min.Y + 12 + i*30
		if y+22 > side.Max.Y {
			break
		}
		textNoise(img, image.Rect(side.Min.X+10, y, side.Max.X-10, y+22), 12, rng)
	}

	// content area: text-like lines (the "panel" template crops from here)
	content := image.Rect(side.Max.X, toolbar.Max.Y, win.Max.X, win.Max.Y)
	textNoise(img, content, 14, rng)
	panelMin := image.Pt(content.Min.X+content.Dx()/4, content.Min.Y+content.Dy()/4)
	panel := image.Rectangle{Min: panelMin, Max: panelMin.Add(image.Pt(300, 200))}

	// dialog with three look-alike buttons; the middle one is the target
	dlg := image.Rect(content.Min.X+content.Dx()/3, content.Min.Y+content.Dy()/2,
		content.Min.X+content.Dx()/3+380, content.Min.Y+content.Dy()/2+150)
	fillRect(img, dlg, color.RGBA{250, 250, 250, 255})
	outline(img, dlg, color.RGBA{80, 80, 80, 255})
	vGradient(img, image.Rect(dlg.Min.X, dlg.Min.Y, dlg.Max.X, dlg.Min.Y+24),
		color.RGBA{235, 235, 235, 255}, color.RGBA{215, 215, 215, 255})
	textNoise(img, image.Rect(dlg.Min.X+16, dlg.Min.Y+36, dlg.Max.X-16, dlg.Min.Y+78), 12, rng)
	var button image.Rectangle
	for i := 0; i < 3; i++ {
		r := image.Rect(dlg.Min.X+24+i*116, dlg.Max.Y-46, dlg.Min.X+24+i*116+96, dlg.Max.Y-14)
		drawButton(img, r, int64(500+i))
		if i == 1 {
			button = r
		}
	}
	return uiTargets{img: img, button: button, icon: icon, panel: panel}
}

// crop copies a region into a tight standalone RGBA template.
func crop(img *image.RGBA, r image.Rectangle) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := 0; y < r.Dy(); y++ {
		copy(out.Pix[y*out.Stride:y*out.Stride+r.Dx()*4],
			img.Pix[img.PixOffset(r.Min.X, r.Min.Y+y):])
	}
	return out
}
