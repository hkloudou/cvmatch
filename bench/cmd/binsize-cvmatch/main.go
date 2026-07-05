// Minimal program used only to measure the linked binary size of cvmatch.
package main

import (
	"fmt"
	"image"

	"github.com/hkloudou/cvmatch"
)

func main() {
	parent := image.NewRGBA(image.Rect(0, 0, 64, 64))
	parent.Pix[0] = 1
	sub := parent.SubImage(image.Rect(0, 0, 8, 8)).(*image.RGBA)
	_, _, _, maxV, maxX, maxY := cvmatch.Match(parent, sub)
	fmt.Println(maxV, maxX, maxY)
}
