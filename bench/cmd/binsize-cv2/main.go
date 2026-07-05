// Minimal program used only to measure the linked binary size of cv2.
package main

import (
	"fmt"
	"image"

	cv2 "github.com/hkloudou/cv2"
)

func main() {
	parent := image.NewRGBA(image.Rect(0, 0, 64, 64))
	parent.Pix[0] = 1
	sub := parent.SubImage(image.Rect(0, 0, 8, 8)).(*image.RGBA)
	_, _, _, maxV, maxX, maxY := cv2.Match(parent, sub)
	fmt.Println(maxV, maxX, maxY)
}
