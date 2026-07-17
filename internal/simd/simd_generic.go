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

func MulConj(spec, tspec []complex64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}

func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
	mean *[4]float64, invArea, eps, templNorm float64) {
	panic("cvmatch: SIMD kernel on unsupported platform")
}
