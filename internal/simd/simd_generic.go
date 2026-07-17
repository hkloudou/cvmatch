//go:build !amd64 || !gc

package simd

// Non-amd64 (or non-gc) builds run the generic Go loops.

// Enabled is false where no assembly kernels exist (a var so tests can
// exercise the SIMD/scalar toggle uniformly across platforms; it must
// never be set true here).
var Enabled = false

func FFTStages(a []complex64, tw []complex64, inverse bool) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func FFTColsBfly(p, q []complex64, w complex64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func FFTCols4(r0, r1, r2, r3 []complex64, w1, w2a, w2b complex64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func MulConj(spec, tspec []complex64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
	mean *[4]float64, invArea, eps, templNorm float64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func PackRows2(z []complex64, ra, rb []uint8, step int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func PackRows1(z []complex64, ra []uint8, step int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func Untangle(sa, sb, z []complex64, n, k0, k1 int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func CombineLow(z, sa, sb []complex64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func CombineHigh(z, sa, sb []complex64, n, hw int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func EmitRe(dst []float32, z []complex64, add bool) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func EmitIm(dst []float32, z []complex64, add bool) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func MinMaxRow(row []float32) (minV, maxV float32, minI, maxI int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func RGBAToGray(dst, src []uint8) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func SlideCols1(colSum []int32, colSum2 []int64, rsub, radd []uint8) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func SlideCols4(colSum []int32, colSum2 []int64, rsub, radd []uint8, cn int) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}
