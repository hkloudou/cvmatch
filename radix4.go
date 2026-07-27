package cvmatch

import (
	"math/bits"

	"github.com/hkloudou/cvmatch/internal/simd"
)

// Radix-4 FFT engine (Phase 7.2), no-FMA — decided by measurement: the
// FMA variant needs the fma32 scalar twin for self-determinism, and
// that measured 3.8x slower in scalar, killing the purego/arm64 leg.
// Both axes run this one engine: the packed row FFTs through fftR4
// (swap pairs applied in the scratch row + FFTStagesR4), the column
// passes through colsR4Go (never swapped — rows are placed at or
// indexed through brev slots) with the FFTColsR4/Head kernels.
//
// Kernel contract (the fixed op sequence the asm must reproduce):
//   - Input is BIT-reversed (the shared makeBitrev table). With plain
//     bit reversal the four quarter blocks of a radix-4 group hold the
//     n%4 == 0,2,1,3 subsequences in that order, so the butterfly reads
//     its q=1 operand at offset 2h and its q=2 operand at offset h.
//   - Complex twiddle multiply t = v*w rounds every product once, no
//     fusion anywhere (mulPlain): tr = f32(vr*wr) - f32(vi*wi),
//     ti = f32(vr*wi) + f32(vi*wr). The asm must use plain VMULPS +
//     VADDPS/VSUBPS in exactly this grouping.
//   - The butterfly combines with plain single-rounded adds in the
//     order written below; the +-i rotation is an exact swap/negate.
//   - Odd log2(n) runs one trivial radix-2 stage (half=1, twiddle 1)
//     first, then radix-4 stages h=2,8,...; even log2(n) starts radix-4
//     at h=1.
//   - Inverse conjugates the twiddles and swaps the +-i rotation; the
//     1/n scale stays the caller's job (unchanged from fftGo).

// oddLog2 reports whether log2(n) is odd (n a power of two), i.e. the
// transform needs the radix-2 head stage.
func oddLog2(n int) bool { return bits.Len(uint(n))%2 == 0 }

// makeTriTwiddles builds the per-stage twiddle triplets for the radix-4
// stages of an n-point transform: for each stage with quarter size h,
// w1[j]=cis(-pi j/(2h)), w2[j]=cis(-pi j/h), w3[j]=cis(-3pi j/(2h)) for
// j<h, stored as three consecutive h-blocks. All values come from the
// shared deterministic sincospiFrac generator; w3 folds 3j into [0,2h)
// with an exact sign flip, so every entry is a directly generated
// dyadic-angle value.
func makeTriTwiddles(n int) []complex64 {
	var tri []complex64
	h := 1
	if oddLog2(n) {
		h = 2
	}
	for ; 4*h <= n; h *= 4 {
		for j := 0; j < h; j++ { // w1 block
			c, s := sincospiFrac(j, 2*h)
			tri = append(tri, complex(c, -s))
		}
		for j := 0; j < h; j++ { // w2 block
			c, s := sincospiFrac(j, h)
			tri = append(tri, complex(c, -s))
		}
		for j := 0; j < h; j++ { // w3 block: 3j folded into [0,2h)
			m, neg := 3*j, false
			if m >= 2*h {
				m, neg = m-2*h, true // cis(-pi(u+1)) = -cis(-pi u), exact
			}
			c, s := sincospiFrac(m, 2*h)
			if neg {
				c, s = -c, -s
			}
			tri = append(tri, complex(c, -s))
		}
	}
	return tri
}

// mulPlain is the engine's complex twiddle multiply: every product and
// sum rounds once in float32, pinned exactly like bfly (the conversions
// stop gc from widening or contracting). An FMA variant (mulFMA via the
// fma32 scalar twin) was built and measured 3.8x slower in scalar —
// dead by the determinism contract's scalar-cost, see CLAUDE.md.
func mulPlain(v, w complex64) complex64 {
	return complex(
		float32(real(v)*real(w))-float32(imag(v)*imag(w)),
		float32(real(v)*imag(w))+float32(imag(v)*real(w)),
	)
}

// fftR4 transforms a in place (len a power of two): bit reversal, the
// optional radix-2 head stage, then radix-4 sweeps. Same external 1/n
// scaling convention as fftGo, so callers swap engines without other
// changes.
func fftR4(a []complex64, tri []complex64, pairs []int32, inverse bool) {
	n := len(a)
	for k := 0; k+1 < len(pairs); k += 2 {
		i, j := pairs[k], pairs[k+1]
		a[i], a[j] = a[j], a[i]
	}
	if n >= 8 && simd.Enabled {
		simd.FFTStagesR4(a, tri, inverse)
		return
	}
	fftR4Scalar(a, tri, inverse)
}

// fftR4Scalar is the engine's scalar cascade (post-swap); the kernel
// parity test pins FFTStagesR4 to it bit-for-bit.
func fftR4Scalar(a []complex64, tri []complex64, inverse bool) {
	n := len(a)
	h := 1
	if oddLog2(n) {
		for i := 0; i < n; i += 2 {
			u, v := a[i], a[i+1]
			a[i] = complex(real(u)+real(v), imag(u)+imag(v))
			a[i+1] = complex(real(u)-real(v), imag(u)-imag(v))
		}
		h = 2
	}
	for ; 4*h <= n; h *= 4 {
		w1, w2, w3 := tri[:h], tri[h:2*h], tri[2*h:3*h]
		tri = tri[3*h:]
		for base := 0; base < n; base += 4 * h {
			for j := 0; j < h; j++ {
				av := a[base+j]
				bw, cw, dw := w1[j], w2[j], w3[j]
				if inverse {
					bw = complex(real(bw), -imag(bw))
					cw = complex(real(cw), -imag(cw))
					dw = complex(real(dw), -imag(dw))
				}
				tb := mulPlain(a[base+j+2*h], bw)
				tc := mulPlain(a[base+j+h], cw)
				td := mulPlain(a[base+j+3*h], dw)
				s0 := complex(real(av)+real(tc), imag(av)+imag(tc))
				s1 := complex(real(av)-real(tc), imag(av)-imag(tc))
				s2 := complex(real(tb)+real(td), imag(tb)+imag(td))
				s3 := complex(real(tb)-real(td), imag(tb)-imag(td))
				var rr, ri float32
				if inverse {
					rr, ri = -imag(s3), real(s3)
				} else {
					rr, ri = imag(s3), -real(s3)
				}
				a[base+j] = complex(real(s0)+real(s2), imag(s0)+imag(s2))
				a[base+j+h] = complex(real(s1)+rr, imag(s1)+ri)
				a[base+j+2*h] = complex(real(s0)-real(s2), imag(s0)-imag(s2))
				a[base+j+3*h] = complex(real(s1)-rr, imag(s1)-ri)
			}
		}
	}
}
