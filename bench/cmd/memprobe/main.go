// memprobe runs one 1080p/128px Match with the chosen implementation and
// prints the process peak RSS (VmHWM), isolating each library's true memory
// footprint in a fresh process.
package main

import (
	"flag"
	"fmt"
	"image"
	"math/rand"
	"os"
	"strings"

	cv2 "github.com/hkloudou/cv2"
	"github.com/hkloudou/cvmatch"
)

func makeImages(w, h, sw, sh, px, py int) (*image.RGBA, *image.RGBA) {
	rng := rand.New(rand.NewSource(1))
	parent := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(parent.Pix); i += 4 {
		parent.Pix[i] = uint8(rng.Intn(256))
		parent.Pix[i+1] = uint8(rng.Intn(256))
		parent.Pix[i+2] = uint8(rng.Intn(256))
		parent.Pix[i+3] = 255
	}
	sub := image.NewRGBA(image.Rect(0, 0, sw, sh))
	for y := 0; y < sh; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+sw*4],
			parent.Pix[(py+y)*parent.Stride+px*4:])
	}
	return parent, sub
}

func vmHWM() string {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "unknown"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			return strings.TrimSpace(strings.TrimPrefix(line, "VmHWM:"))
		}
	}
	return "unknown"
}

func main() {
	impl := flag.String("impl", "cvmatch", "cv2 | cvmatch | gray | baseline")
	flag.Parse()

	parent, sub := makeImages(1920, 1080, 128, 128, 977, 604)
	base := vmHWM()

	var maxV float32
	var maxX, maxY int
	switch *impl {
	case "cv2":
		_, _, _, maxV, maxX, maxY = cv2.Match(parent, sub)
	case "cvmatch":
		_, _, _, maxV, maxX, maxY = cvmatch.Match(parent, sub)
	case "gray":
		_, _, _, maxV, maxX, maxY = cvmatch.MatchGray(parent, sub)
	case "baseline":
		// images allocated, no matching: isolates the match cost.
	}
	fmt.Printf("impl=%s match=(%d,%d val=%.4f) baselineHWM=%s peakHWM=%s\n",
		*impl, maxX, maxY, maxV, base, vmHWM())
}
