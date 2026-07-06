// annotate runs cvmatch.Match on selected scenes and renders the result:
// the parent image with a box drawn at the found best-match location, plus
// the template saved alongside. The outputs are published on the orphan
// assets branch, which the README references to show what "find the best
// match in an image" looks like.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/hkloudou/cvmatch"
	"github.com/hkloudou/cvmatch/scenes"
)

func drawRect(img *image.RGBA, r image.Rectangle, c color.RGBA, thick int) {
	for t := 0; t < thick; t++ {
		rr := image.Rect(r.Min.X-t, r.Min.Y-t, r.Max.X+t, r.Max.Y+t).Intersect(img.Bounds())
		for x := rr.Min.X; x < rr.Max.X; x++ {
			img.SetRGBA(x, rr.Min.Y, c)
			img.SetRGBA(x, rr.Max.Y-1, c)
		}
		for y := rr.Min.Y; y < rr.Max.Y; y++ {
			img.SetRGBA(rr.Min.X, y, c)
			img.SetRGBA(rr.Max.X-1, y, c)
		}
	}
}

func save(path string, img image.Image, asJPEG bool) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	if asJPEG {
		err = jpeg.Encode(f, img, &jpeg.Options{Quality: 85})
	} else {
		err = png.Encode(f, img)
	}
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	out := flag.String("out", "demo-out", "output directory")
	flag.Parse()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	want := map[string]bool{
		"window1600_button96x32": true, "photo_fruits": true, "photo_baboon": true,
		"photo_building": true, "photo_graf1": true, "photo_starry_night": true,
		"noise640_alpha": true, // saved as PNG: JPEG would discard the alpha plane
	}
	green := color.RGBA{0, 224, 64, 255}
	for _, s := range scenes.All("testdata") {
		if !want[s.Name] {
			continue
		}
		_, _, _, maxV, maxX, maxY := cvmatch.Match(s.Parent, s.Sub)
		annotated := image.NewRGBA(s.Parent.Bounds())
		copy(annotated.Pix, s.Parent.Pix)
		sb := s.Sub.Bounds()
		drawRect(annotated, image.Rect(maxX, maxY, maxX+sb.Dx(), maxY+sb.Dy()), green, 3)
		if s.Name == "noise640_alpha" {
			save(filepath.Join(*out, s.Name+".png"), annotated, false)
		} else {
			save(filepath.Join(*out, s.Name+".jpg"), annotated, true)
		}
		save(filepath.Join(*out, s.Name+".tpl.png"), s.Sub, false)
		fmt.Printf("%s\tmaxVal=%.6f\tloc=(%d,%d)\n", s.Name, maxV, maxX, maxY)
	}
}
