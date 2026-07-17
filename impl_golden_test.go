package cvmatch

import (
	"hash/fnv"
	"math"
	"math/rand"
	"testing"
)

// goldenHash folds a response map and its extrema into one FNV-1a 64-bit
// value over the exact float32 bit patterns (little-endian), so a single
// constant pins every output bit of a match call.
func goldenHash(res []float32, minV float32, minX, minY int, maxV float32, maxX, maxY int) uint64 {
	h := fnv.New64a()
	var b [4]byte
	put := func(bits uint32) {
		b[0], b[1], b[2], b[3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
		h.Write(b[:])
	}
	for _, v := range res {
		put(math.Float32bits(v))
	}
	put(math.Float32bits(minV))
	put(uint32(minX))
	put(uint32(minY))
	put(math.Float32bits(maxV))
	put(uint32(maxX))
	put(uint32(maxY))
	return h.Sum64()
}

// goldenCases pins one deterministic input per dispatch shape. The hashes
// were recorded from the cgo C core immediately before its removal, with
// the pure-Go core verified bit-identical at recording time; the C core
// itself was pinned element-wise against native OpenCV (the bench/ parity
// jobs still are). Together with TestAgainstReference they anchor the
// OpenCV bit-identity guarantee for all future changes.
var goldenCases = []struct {
	seed           int64
	iw, ih, tw, th int
	cn, step       int
	ipad, tpad     int // extra row padding in pixels (strided sub-image case)
	hash           uint64
}{
	{101, 97, 61, 17, 13, 1, 1, 0, 0, 0x23cc16bf44c26385},   // grayscale
	{102, 97, 61, 17, 13, 4, 4, 0, 0, 0x433a9244109eab94},   // full RGBA
	{103, 120, 90, 24, 18, 3, 4, 0, 0, 0xaabe11c6793aced6},  // RGBA with alpha skipped (cn=3, step=4)
	{104, 300, 200, 31, 29, 1, 1, 7, 3, 0xe29430483f01d531}, // strided rows (sub-image)
	{105, 64, 64, 64, 64, 1, 1, 0, 0, 0xbd496a1d2849c25d},   // template == image (1x1 result)
	{106, 50, 40, 1, 1, 4, 4, 0, 0, 0x7711e78701385195},     // 1x1 template
	{107, 640, 400, 96, 32, 3, 4, 0, 0, 0x773d6b8576272893}, // multi-block FFT tiling
	{108, 128, 96, 5, 4, 3, 3, 0, 0, 0xaa868693616d67ec},    // packed RGB (step==cn=3, scalar pack fallback)
	{109, 40, 300, 33, 257, 1, 1, 0, 0, 0xf573e86e02737f87}, // tall template, tiny result width
}

func goldenInput(c struct {
	seed           int64
	iw, ih, tw, th int
	cn, step       int
	ipad, tpad     int
	hash           uint64
}) (img, tpl []uint8, istride, tstride int) {
	rng := rand.New(rand.NewSource(c.seed))
	istride, tstride = (c.iw+c.ipad)*c.step, (c.tw+c.tpad)*c.step
	return randPix(istride*c.ih, rng), randPix(tstride*c.th, rng), istride, tstride
}

// TestGoldenOutputs asserts the core reproduces the recorded output bits —
// map and extrema — for every dispatch shape, single- and multi-threaded.
// The same constants must hold on every architecture (SIMD kernels and the
// scalar fallbacks execute identical single-rounded IEEE op sequences).
func TestGoldenOutputs(t *testing.T) {
	for ci, c := range goldenCases {
		img, tpl, istride, tstride := goldenInput(c)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1
		for _, nt := range []int{1, 4} {
			res := make([]float32, rw*rh)
			minV, minX, minY, maxV, maxX, maxY := matchU8(img, istride, c.iw, c.ih,
				tpl, tstride, c.tw, c.th, c.cn, c.step, nt, res)
			if got := goldenHash(res, minV, minX, minY, maxV, maxX, maxY); got != c.hash {
				t.Errorf("case %d (threads=%d): output hash %#016x, want recorded %#016x",
					ci, nt, got, c.hash)
			}
		}
	}
}
