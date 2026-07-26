package cvmatch

import (
	"math"
	"testing"
)

// The column twin must apply exactly the row engine's op sequence to
// every column: outputs are compared bit-for-bit (not by tolerance),
// across widths that exercise chunk tails and for team counts that
// exercise the width-split scheduling.
func TestColsR4MatchesRowEngine(t *testing.T) {
	for _, n := range []int{2, 4, 8, 16, 64, 128, 512} {
		tab := makeR4Tab(n)
		for _, width := range []int{1, 7, 8, 24, 33} {
			src := randSlab(n*width, int64(n*width))
			want := make([]complex64, n*width)
			copy(want, src)
			col := make([]complex64, n)
			for _, inverse := range []bool{false, true} {
				for c := 0; c < width; c++ {
					for r := 0; r < n; r++ {
						col[r] = src[r*width+c]
					}
					fftR4(col, tab, inverse)
					for r := 0; r < n; r++ {
						want[r*width+c] = col[r]
					}
				}
				for _, team := range []int{1, 3} {
					got := make([]complex64, n*width)
					copy(got, src)
					tmp := make([]complex64, width)
					colsR4Go(got, n, width, tab, inverse, tmp, team)
					for i := range got {
						if math.Float32bits(real(got[i])) != math.Float32bits(real(want[i])) ||
							math.Float32bits(imag(got[i])) != math.Float32bits(imag(want[i])) {
							t.Fatalf("n=%d width=%d inverse=%v team=%d elem %d: got %v want %v (bit mismatch)",
								n, width, inverse, team, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// Cost probe: the current column engine vs the radix-4 twin on a
// pipeline-shaped slab (n=512 columns of a 256-wide spectrum).
func BenchmarkColsRadix2(b *testing.B) {
	const n, width = 512, 256
	ft := fftTables(n)
	d := randSlab(n*width, 9)
	tmp := make([]complex64, width)
	b.SetBytes(int64(n * width * 8))
	for i := 0; i < b.N; i++ {
		fftColsGo(d, n, width, ft.tw, ft.pairs, i&1 == 1, tmp, 1)
	}
}

func BenchmarkColsRadix4(b *testing.B) {
	const n, width = 512, 256
	tab := makeR4Tab(n)
	d := randSlab(n*width, 9)
	tmp := make([]complex64, width)
	b.SetBytes(int64(n * width * 8))
	for i := 0; i < b.N; i++ {
		colsR4Go(d, n, width, tab, i&1 == 1, tmp, 1)
	}
}
