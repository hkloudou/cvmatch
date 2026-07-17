//go:build gc

package simd

// AVX2 kernels for the pure-Go core's hot loops. Each vector lane executes
// exactly the scalar op sequence (individual IEEE multiplies and adds, no
// FMA — the gc compiler does not contract on amd64 either), so results are
// bit-identical to the generic Go loops on every input; the asm is a pure
// speedup, not a numeric variant. Detection requires both CPU support and
// OS-enabled YMM state.

// Enabled reports whether the AVX2 kernels can run (CPU + OS support).
var Enabled = detectAVX2()

func detectAVX2() bool {
	maxID, _, _, _ := cpuid(0, 0)
	if maxID < 7 {
		return false
	}
	_, _, c1, _ := cpuid(1, 0)
	const osxsaveAVX = 1<<27 | 1<<28
	if c1&osxsaveAVX != osxsaveAVX {
		return false
	}
	if xgetbv()&6 != 6 { // OS saves XMM+YMM state
		return false
	}
	_, b7, _, _ := cpuid(7, 0)
	return b7&(1<<5) != 0 // AVX2
}

func cpuid(leaf, sub uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() uint64

// FFTStages runs every radix-2 stage with half >= 4 over a (natural
// bit-reversed layout, len(a) a power of two >= 8).
//
//go:noescape
func FFTStages(a []complex64, tw []complex64, inverse bool)

// FFTColsBfly applies one column-FFT butterfly row pair with the
// (already sign-adjusted) twiddle w: p,q = p+q*w, p-q*w.
//
//go:noescape
func FFTColsBfly(p, q []complex64, w complex64)

// MulConj computes spec *= conj(tspec) element-wise.
//
//go:noescape
func MulConj(spec, tspec []complex64)

// NormRow evaluates the TM_CCOEFF_NORMED tail over one chunk of n result
// elements. wt holds cn lanes of window sums followed by one lane of
// window square sums, each stride float64s apart, all exact integers.
// Every vector lane executes the scalar op sequence (vsqrtpd/vdivpd are
// correctly rounded); n must be a multiple of 4 (the caller finishes the
// tail in Go). cn must be 1, 3 or 4.
//
//go:noescape
func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
	mean *[4]float64, invArea, eps, templNorm float64)
