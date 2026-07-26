//go:build purego || !amd64 || !gc

package simd

// Builds without assembly kernels: -tags purego opts out anywhere, and
// every non-amd64 architecture (arm64 included) lands here in every
// build mode. The scalar Go loops produce bit-identical output — the
// golden anchors reprove it on the arm64 CI leg.

// Enabled is a constant false here, so kernel call sites dead-code
// eliminate out of default builds entirely.
const Enabled = false

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

func NormRow(rrow []float32, crow []float32, wt *float32, stride, n int,
	numScale, varScale, eps, templNorm float32) {
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

func SpillStats1(wt []float32, stride int, lo, hi []int32, lo2, hi2 []int64, s0, s2, area int64) (ns0, ns2 int64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}
