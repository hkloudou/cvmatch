package cvmatch

import (
	"math"
	"testing"
)

// The column engine must apply exactly the row engine's op sequence to
// every column — outputs are compared bit-for-bit (not by tolerance) —
// in both storage modes: rmap == nil consumes input whose logical row L
// sits at slot brev[L] and leaves output natural (the forward path:
// writers pre-place rows), rmap = brev consumes natural input and
// leaves the transform of logical slot s at physical row brev[s] (the
// inverse path: MulConj output feeds it unmoved). Widths exercise chunk
// tails, team counts the width-split scheduling.
func TestColsR4MatchesRowEngine(t *testing.T) {
	for _, n := range []int{2, 4, 8, 16, 64, 128, 512} {
		ft := fftTables(n)
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
					fftR4(col, ft.tri(), ft.pairs, inverse)
					for r := 0; r < n; r++ {
						want[r*width+c] = col[r]
					}
				}
				for _, team := range []int{1, 3} {
					// rmap == nil: pre-place logical row L at slot brev[L],
					// expect natural output.
					got := make([]complex64, n*width)
					for r := 0; r < n; r++ {
						copy(got[int(ft.brev[r])*width:][:width], src[r*width:][:width])
					}
					colsR4Go(got, n, width, ft.tri(), inverse, nil, team)
					for i := range got {
						if math.Float32bits(real(got[i])) != math.Float32bits(real(want[i])) ||
							math.Float32bits(imag(got[i])) != math.Float32bits(imag(want[i])) {
							t.Fatalf("nil rmap: n=%d width=%d inverse=%v team=%d elem %d: got %v want %v",
								n, width, inverse, team, i, got[i], want[i])
						}
					}
					// rmap = brev: natural input, output row s at slot brev[s].
					got2 := make([]complex64, n*width)
					copy(got2, src)
					colsR4Go(got2, n, width, ft.tri(), inverse, ft.brev, team)
					for r := 0; r < n; r++ {
						b := int(ft.brev[r]) * width
						for c := 0; c < width; c++ {
							g, w := got2[b+c], want[r*width+c]
							if math.Float32bits(real(g)) != math.Float32bits(real(w)) ||
								math.Float32bits(imag(g)) != math.Float32bits(imag(w)) {
								t.Fatalf("brev rmap: n=%d width=%d inverse=%v team=%d row %d col %d: got %v want %v",
									n, width, inverse, team, r, c, g, w)
							}
						}
					}
				}
			}
		}
	}
}

// Standing cost probe for the column engine on a pipeline-shaped slab
// (n=512 columns of a 256-wide spectrum). At the 7.2 flip this measured
// +14.5% asm / +18.7% purego over the retired radix-2 column passes;
// since the Phase 9 swap removal it times the cascade alone (the
// pipeline no longer runs any row-swap sweep).
func BenchmarkColsRadix4(b *testing.B) {
	const n, width = 512, 256
	ft := fftTables(n)
	d := randSlab(n*width, 9)
	b.SetBytes(int64(n * width * 8))
	for i := 0; i < b.N; i++ {
		colsR4Go(d, n, width, ft.tri(), i&1 == 1, ft.brev, 1)
	}
}
