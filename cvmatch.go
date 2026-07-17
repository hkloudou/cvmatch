// Package cvmatch implements OpenCV-compatible TM_CCOEFF_NORMED template
// matching in pure Go — no OpenCV, no cgo, no dependencies. Hand-written
// SIMD kernels (AVX2 on amd64, NEON on arm64) accelerate the hot loops;
// every other platform runs the scalar Go code with identical output.
//
// Match is numerically aligned with OpenCV's matchTemplate + minMaxLoc on
// CV_8UC4 input (the classic ImageToMatRGBA-style pipeline), while
// MatchGray trades exact RGBA parity for ~4x less work. Output is
// bit-identical across architectures, thread counts and SIMD on/off: the
// pipeline runs one fixed single-rounded IEEE op sequence, including a
// deterministic twiddle generator independent of the system libm, and the
// golden tests pin the exact bits.
package cvmatch

import (
	"image"
	"image/draw"
	"os"
	"runtime"
	"strconv"

	"github.com/hkloudou/cvmatch/internal/simd"
)

// Impl reports the active core. Since the cgo/C core was retired (the Go
// core benchmarks ahead of it on both amd64 and arm64) there is only one:
// "purego". Kept for compatibility.
const Impl = "purego"

// Threads overrides the number of workers a single Match/MatchMap/MatchGray
// call uses internally. 0 (the default) means automatic: GOMAXPROCS capped
// at 16. Any other value is clamped to [1, 16]. Every setting produces
// bit-identical output (asserted by TestThreadsBitIdentical*), so this is a
// performance/benchmarking knob only — e.g. Threads = 1 pins single-threaded
// behavior. It can also be set through the CVMATCH_THREADS environment
// variable, read once at package init. Set it before concurrent use begins;
// it is a plain variable and is not synchronized.
var Threads int

func init() {
	if n, err := strconv.Atoi(os.Getenv("CVMATCH_THREADS")); err == nil && n > 0 {
		Threads = n
	}
}

// threads resolves Threads into the worker count for one call.
func threads() int {
	n := Threads
	if n <= 0 {
		n = runtime.GOMAXPROCS(0)
	}
	if n > 16 {
		n = 16
	}
	if n < 1 {
		n = 1
	}
	return n
}

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
	return matchU8(pPix, pStride, pw, ph, sPix, sStride, sw, sh, cn, 4, threads(), nil)
}

// MatchMap runs the same computation as Match but returns the full
// normalized response map (row-major, (parentW-subW+1) x (parentH-subH+1)),
// equivalent to OpenCV's matchTemplate output. Useful for finding every
// occurrence above a threshold instead of only the best one.
func MatchMap(parent, sub image.Image) (res []float32, w, h int) {
	pPix, pStride, pw, ph := toRGBA(parent)
	sPix, sStride, sw, sh := toRGBA(sub)
	cn := 4
	if alphaConst(pPix, pStride, pw, ph) && alphaConst(sPix, sStride, sw, sh) {
		cn = 3
	}
	w, h = pw-sw+1, ph-sh+1
	res = make([]float32, w*h)
	matchU8(pPix, pStride, pw, ph, sPix, sStride, sw, sh, cn, 4, threads(), res)
	return res, w, h
}

// MatchGray is the fast path: both images are reduced to 8-bit grayscale
// (OpenCV BT.601 RGB2GRAY weights; the Y plane is used directly for YCbCr
// images) before matching. For RGB screenshots the response map is very
// close to the 4-channel one but costs roughly a quarter of the work.
func MatchGray(parent, sub image.Image) (float32, int, int, float32, int, int) {
	pPix, pStride, pw, ph, pOwned := toGray(parent)
	sPix, sStride, sw, sh, sOwned := toGray(sub)
	r1, r2, r3, r4, r5, r6 := matchU8(pPix, pStride, pw, ph, sPix, sStride, sw, sh, 1, 1, threads(), nil)
	if pOwned {
		bytePool.put(pPix)
	}
	if sOwned {
		bytePool.put(sPix)
	}
	return r1, r2, r3, r4, r5, r6
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
// in place with zero copies (sub-images included — the core honors the
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
// NRGBA are converted with OpenCV's fixed-point BT.601 weights into a
// pooled buffer (owned reports it so the caller can recycle it).
func toGray(img image.Image) (pix []uint8, stride, w, h int, owned bool) {
	b := bounds(img)
	w, h = b.Dx(), b.Dy()
	switch m := img.(type) {
	case *image.Gray:
		return m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h, false
	case *image.YCbCr:
		return m.Y[m.YOffset(b.Min.X, b.Min.Y):], m.YStride, w, h, false
	case *image.RGBA:
		return rgbToGray(m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h), w, w, h, true
	case *image.NRGBA:
		return rgbToGray(m.Pix[m.PixOffset(b.Min.X, b.Min.Y):], m.Stride, w, h), w, w, h, true
	}
	gray := image.NewGray(image.Rect(0, 0, w, h))
	draw.Draw(gray, gray.Bounds(), img, b.Min, draw.Src)
	return gray.Pix, gray.Stride, w, h, false
}

// rgbToGray converts interleaved 4-byte pixels using OpenCV's RGB2GRAY
// fixed-point coefficients: (4899*R + 9617*G + 1868*B + 8192) >> 14.
// Integer arithmetic — the result is exact regardless of evaluation order.
func rgbToGray(pix []uint8, stride, w, h int) []uint8 {
	out := bytePool.get(w * h)
	vw := 0
	if simd.Enabled {
		vw = w &^ 7
	}
	for y := 0; y < h; y++ {
		src := pix[y*stride : y*stride+w*4]
		dst := out[y*w : y*w+w]
		if vw > 0 {
			simd.RGBAToGray(dst[:vw], src)
		}
		for x := vw; x < w; x++ {
			r, g, b := uint32(src[x*4]), uint32(src[x*4+1]), uint32(src[x*4+2])
			dst[x] = uint8((4899*r + 9617*g + 1868*b + 8192) >> 14)
		}
	}
	return out
}
