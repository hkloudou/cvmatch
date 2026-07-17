//go:build cvmatch_asm

package cvmatch

// Kernel-only tests: they exercise the assembly directly (and toggle
// simd.Enabled, which is a constant in default builds), so the whole file
// rides the cvmatch_asm tag.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/hkloudou/cvmatch/internal/simd"
)

// TestSIMDMatchesScalar proves the assembly kernels are a pure speedup:
// with simd.Enabled toggled off, the generic Go loops must produce
// bit-identical response maps and extrema. (The kernels perform exactly
// the scalar single-rounded op sequence per lane, so any deviation is a
// bug.) On platforms without kernels the test degenerates to
// scalar==scalar and is skipped.
func TestSIMDMatchesScalar(t *testing.T) {
	if !simd.Enabled {
		t.Skip("no SIMD kernels on this platform")
	}
	defer func() { simd.Enabled = true }()

	rng := rand.New(rand.NewSource(123))
	cases := []struct{ iw, ih, tw, th, cn, step int }{
		{97, 61, 17, 13, 1, 1},
		{120, 90, 24, 18, 3, 4},
		{640, 400, 96, 32, 3, 4},
		{300, 200, 31, 29, 4, 4},
		{128, 96, 5, 4, 3, 3},    // packed RGB: pack kernels must stand down
		{50, 40, 1, 1, 4, 4},     // 1x1 template, tiny DFT sizes
		{40, 300, 33, 257, 1, 1}, // tall template, tiny rw
	}
	for ci, c := range cases {
		img := randPix(c.iw*c.ih*c.step, rng)
		tpl := randPix(c.tw*c.th*c.step, rng)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1

		simd.Enabled = true
		fast := make([]float32, rw*rh)
		fMinV, fMinX, fMinY, fMaxV, fMaxX, fMaxY := matchU8(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 3, fast)

		simd.Enabled = false
		slow := make([]float32, rw*rh)
		sMinV, sMinX, sMinY, sMaxV, sMaxX, sMaxY := matchU8(img, c.iw*c.step, c.iw, c.ih, tpl, c.tw*c.step, c.tw, c.th, c.cn, c.step, 3, slow)

		for i := range fast {
			if math.Float32bits(fast[i]) != math.Float32bits(slow[i]) {
				t.Fatalf("case %d: SIMD map differs from scalar at %d: %v vs %v", ci, i, fast[i], slow[i])
			}
		}
		if fMinX != sMinX || fMinY != sMinY || fMaxX != sMaxX || fMaxY != sMaxY ||
			math.Float32bits(fMinV) != math.Float32bits(sMinV) ||
			math.Float32bits(fMaxV) != math.Float32bits(sMaxV) {
			t.Fatalf("case %d: SIMD extrema differ from scalar", ci)
		}
	}
}

// TestMinMaxRowKernel exercises the first-occurrence contract directly:
// ties must resolve to the lowest index whichever vector lane they land
// in, matching a scalar strict-compare scan.
func TestMinMaxRowKernel(t *testing.T) {
	if !simd.Enabled {
		t.Skip("no SIMD kernels on this platform")
	}
	rng := rand.New(rand.NewSource(7))
	rows := [][]float32{
		make([]float32, 8),    // all zero: ties everywhere
		make([]float32, 4096), // all zero, long
	}
	for _, n := range []int{8, 16, 64, 200 &^ 7, 4096} {
		r := make([]float32, n)
		for i := range r {
			r[i] = float32(rng.Intn(9)) // few distinct values: dense ties
		}
		rows = append(rows, r)
		r2 := make([]float32, n)
		for i := range r2 {
			r2[i] = rng.Float32()
		}
		rows = append(rows, r2)
	}
	// Extrema in every lane position, duplicated at a later position.
	for k := 0; k < 8; k++ {
		r := make([]float32, 64)
		for i := range r {
			r[i] = 0.5
		}
		r[8+k] = -3
		r[40+(k+3)%8] = -3
		r[16+k] = 7
		r[48+(k+5)%8] = 7
		rows = append(rows, r)
	}
	for ri, r := range rows {
		gMinV, gMaxV, gMinI, gMaxI := simd.MinMaxRow(r)
		sMinV, sMaxV := r[0], r[0]
		sMinI, sMaxI := 0, 0
		for i, v := range r {
			if v < sMinV {
				sMinV, sMinI = v, i
			}
			if v > sMaxV {
				sMaxV, sMaxI = v, i
			}
		}
		if math.Float32bits(gMinV) != math.Float32bits(sMinV) || gMinI != sMinI ||
			math.Float32bits(gMaxV) != math.Float32bits(sMaxV) || gMaxI != sMaxI {
			t.Fatalf("row %d: kernel (%v@%d, %v@%d) vs scalar (%v@%d, %v@%d)",
				ri, gMinV, gMinI, gMaxV, gMaxI, sMinV, sMinI, sMaxV, sMaxI)
		}
	}
}

// TestRGBAToGrayKernel checks the fixed-point conversion (via the
// rgbToGray wrapper: kernel body plus scalar tail) against the scalar
// expression, over random pixels, saturating extremes, and odd widths
// with padded strides.
func TestRGBAToGrayKernel(t *testing.T) {
	if !simd.Enabled {
		t.Skip("no SIMD kernels on this platform")
	}
	rng := rand.New(rand.NewSource(8))
	for _, c := range []struct{ w, h, pad int }{
		{1, 1, 0}, {7, 3, 0}, {8, 2, 4}, {33, 5, 8}, {256, 4, 0}, {641, 2, 12},
	} {
		stride := c.w*4 + c.pad
		pix := randPix(stride*c.h, rng)
		for i := 0; i < len(pix) && i < 64; i += 4 { // saturating extremes
			pix[i], pix[i+1], pix[i+2] = 255, 255, 255
		}
		got := rgbToGray(pix, stride, c.w, c.h)
		for y := 0; y < c.h; y++ {
			for x := 0; x < c.w; x++ {
				p := pix[y*stride+x*4:]
				want := uint8((4899*uint32(p[0]) + 9617*uint32(p[1]) + 1868*uint32(p[2]) + 8192) >> 14)
				if got[y*c.w+x] != want {
					t.Fatalf("w=%d pad=%d (%d,%d): got %d want %d", c.w, c.pad, x, y, got[y*c.w+x], want)
				}
			}
		}
		bytePool.put(got)
	}
}
