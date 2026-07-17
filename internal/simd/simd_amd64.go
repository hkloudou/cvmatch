//go:build gc && cvmatch_asm

package simd

// AVX2 backing for the kernels declared in simd_kernels.go. Detection
// requires both CPU support and OS-enabled YMM state.

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
