// Package cvmatch implements OpenCV-compatible TM_CCOEFF_NORMED template
// matching as a single self-contained cgo file — no OpenCV, no third-party
// static libraries, no dependencies.
//
// Match is numerically aligned with gocv/hkloudou-cv2 style
// Match(parent, sub) built on ImageToMatRGBA + MatchTemplate + MinMaxLoc,
// while MatchGray trades exact RGBA parity for ~4x less work.
package cvmatch

/*
#cgo CFLAGS: -O3 -std=c99
#cgo LDFLAGS: -lm
#include "cvmatch.h"
*/
import "C"

import (
	"fmt"
	"image"
	"image/draw"
	"unsafe"
)

// Match looks for sub inside parent using normalized correlation coefficient
// template matching (TM_CCOEFF_NORMED), processing images as 8-bit RGBA
// exactly like OpenCV's matchTemplate on CV_8UC4 input.
//
// It returns, in order: the minimum match value and its X/Y position, then
// the maximum match value and its X/Y position. The best candidate is the
// maximum: a maxVal close to 1.0 means a confident match whose top-left
// corner is at (maxX, maxY) in parent coordinates.
//
// Match panics if an image is empty or if sub is larger than parent.
func Match(parent, sub image.Image) (float32, int, int, float32, int, int) {
	pPix, pStride, pw, ph := toRGBA(parent)
	sPix, sStride, sw, sh := toRGBA(sub)
	// A channel that is constant within each image (almost always the alpha
	// plane of screenshots) contributes exactly zero to the CCOEFF_NORMED
	// numerator, denominator and template norm, so skipping it changes
	// nothing in the output but saves a quarter of the work.
	cn := 4
	if alphaConst(pPix, pStride, pw, ph) && alphaConst(sPix, sStride, sw, sh) {
		cn = 3
	}
	return matchU8(pPix, pStride, pw, ph, sPix, sStride, sw, sh, cn, 4, nil)
}

// MatchGray is the fast path: both images are reduced to 8-bit grayscale
// (OpenCV BT.601 RGB2GRAY weights; the Y plane is used directly for YCbCr
// images) before matching. For RGB screenshots the response map is very
// close to the 4-channel one but costs roughly a quarter of the work.
func MatchGray(parent, sub image.Image) (float32, int, int, float32, int, int) {
	pPix, pStride, pw, ph := toGray(parent)
	sPix, sStride, sw, sh := toGray(sub)
	return matchU8(pPix, pStride, pw, ph, sPix, sStride, sw, sh, 1, 1, nil)
}

// alphaConst reports whether the alpha plane of interleaved RGBA pixels is a
// single constant value.
func alphaConst(pix []uint8, stride, w, h int) bool {
	a := pix[3]
	for y := 0; y < h; y++ {
		row := pix[y*stride : y*stride+w*4]
		for x := 3; x < len(row); x += 4 {
			if row[x] != a {
				return false
			}
		}
	}
	return true
}

func matchU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step int, result []float32) (float32, int, int, float32, int, int) {
	if tw > iw || th > ih {
		panic(fmt.Sprintf("cvmatch: template %dx%d larger than image %dx%d", tw, th, iw, ih))
	}
	var resPtr *C.float
	if result != nil {
		resPtr = (*C.float)(unsafe.Pointer(&result[0]))
	}
	var out C.CvmExtrema
	rc := C.cvm_match_ccoeff_normed_u8(
		(*C.uint8_t)(unsafe.Pointer(&img[0])), C.int(istride), C.int(iw), C.int(ih),
		(*C.uint8_t)(unsafe.Pointer(&tpl[0])), C.int(tstride), C.int(tw), C.int(th),
		C.int(cn), C.int(step), resPtr, &out)
	if rc != C.CVM_OK {
		panic(fmt.Sprintf("cvmatch: match failed (code %d)", int(rc)))
	}
	return float32(out.min_val), int(out.min_x), int(out.min_y),
		float32(out.max_val), int(out.max_x), int(out.max_y)
}

func bounds(img image.Image) image.Rectangle {
	if img == nil {
		panic("cvmatch: nil image")
	}
	b := img.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		panic("cvmatch: empty image")
	}
	return b
}

// toRGBA returns interleaved RGBA pixels plus stride. A *image.RGBA is used
// in place with zero copies (sub-images included — the C side honors the
// stride); everything else is redrawn, matching ImageToMatRGBA semantics.
func toRGBA(img image.Image) (pix []uint8, stride, w, h int) {
	b := bounds(img)
	w, h = b.Dx(), b.Dy()
	if m, ok := img.(*image.RGBA); ok {
		return m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h
	}
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)
	return rgba.Pix, rgba.Stride, w, h
}

// toGray returns single-channel 8-bit pixels plus stride. *image.Gray and
// the Y plane of *image.YCbCr are used in place with zero copies; RGBA and
// NRGBA are converted with OpenCV's fixed-point BT.601 weights.
func toGray(img image.Image) (pix []uint8, stride, w, h int) {
	b := bounds(img)
	w, h = b.Dx(), b.Dy()
	switch m := img.(type) {
	case *image.Gray:
		return m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h
	case *image.YCbCr:
		return m.Y[m.YOffset(b.Min.X, b.Min.Y):], m.YStride, w, h
	case *image.RGBA:
		return rgbToGray(m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h), w, w, h
	case *image.NRGBA:
		return rgbToGray(m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h), w, w, h
	}
	gray := image.NewGray(image.Rect(0, 0, w, h))
	draw.Draw(gray, gray.Bounds(), img, b.Min, draw.Src)
	return gray.Pix, gray.Stride, w, h
}

// rgbToGray converts interleaved 4-byte pixels using OpenCV's RGB2GRAY
// fixed-point coefficients: (4899*R + 9617*G + 1868*B + 8192) >> 14.
func rgbToGray(pix []uint8, stride, w, h int) []uint8 {
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		src := pix[y*stride:]
		dst := out[y*w:]
		for x := 0; x < w; x++ {
			r, g, b := uint32(src[x*4]), uint32(src[x*4+1]), uint32(src[x*4+2])
			dst[x] = uint8((4899*r + 9617*g + 1868*b + 8192) >> 14)
		}
	}
	return out
}
