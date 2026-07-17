//go:build gc

#include "textflag.h"

// AVX2 kernels: bit-identical to the generic Go loops (see
// impl_simd_amd64.go). Complex64 values are interleaved (re, im) float32
// pairs; the butterfly evaluates t1 = q*wr, t2 = swap(q)*wi and
// addsub(t1, t2) = (t1e-t2e, t1o+t2o), which reproduces the scalar
// complex multiply's two products and single rounded add/sub per
// component (the imaginary sum is commuted; IEEE addition commutes
// bit-exactly).

// func cpuid(leaf, sub uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL sub+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() uint64
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	XORL CX, CX
	XGETBV
	SHLQ $32, DX
	ORQ  DX, AX
	MOVQ AX, ret+0(FP)
	RET

// func fftStagesSIMD(a []complex64, tw []complex64, inverse bool)
TEXT ·FFTStages(SB), NOSPLIT, $0-49
	MOVQ    a_base+0(FP), DI
	MOVQ    a_len+8(FP), SI
	MOVQ    tw_base+24(FP), DX
	MOVBLZX inverse+48(FP), AX
	SHLL    $31, AX             // 0x80000000 when inverse, 0 otherwise
	MOVD    AX, X0
	VPBROADCASTD X0, Y15

	// Stages half=1 and half=2 fused in one pass: every 4-complex group
	// is closed under both stages, so each YMM makes one memory round
	// trip for two butterfly layers. The twiddles are multiplied exactly
	// like the generic loop (no 1/-i shortcuts), so the op sequence per
	// element is untouched.
	VBROADCASTSS 8(DX), Y12     // re(tw[1])
	VBROADCASTSS 12(DX), Y13    // im(tw[1])
	VXORPS       Y15, Y13, Y13  // s*wi
	VMOVUPS      16(DX), X14    // tw[2], tw[3]
	VMOVSLDUP    X14, X0        // (w2r, w2r, w3r, w3r)
	VMOVSHDUP    X14, X14       // (w2i, w2i, w3i, w3i)
	VINSERTF128  $1, X0, Y0, Y0
	VINSERTF128  $1, X14, Y14, Y14
	VXORPS       Y15, Y14, Y14  // s*wi
	XORQ         R10, R10

fs12_loop:
	VMOVUPS   (DI)(R10*8), Y1

	// half=1: butterflies (c0,c1) and (c2,c3), twiddle tw[1]
	VPERMILPS $0xB1, Y1, Y5     // re/im swapped
	VMULPS    Y12, Y1, Y6       // t1 = c*wr
	VMULPS    Y13, Y5, Y7       // t2 = swap(c)*s*wi
	VADDSUBPS Y7, Y6, Y8        // q*w in the odd complex lanes
	VPERMILPD $0x0F, Y8, Y8     // (qw1, qw1, qw3, qw3)
	VMOVDDUP  Y1, Y9            // (c0, c0, c2, c2)
	VADDPS    Y8, Y9, Y10       // p + qw
	VSUBPS    Y8, Y9, Y11       // p - qw
	VBLENDPS  $0xCC, Y11, Y10, Y1

	// half=2: butterflies (c0,c2) and (c1,c3), twiddles tw[2], tw[3]
	VPERMILPS  $0xB1, Y1, Y5
	VMULPS     Y0, Y1, Y6
	VMULPS     Y14, Y5, Y7
	VADDSUBPS  Y7, Y6, Y8       // q*w in the high 128-bit lane
	VPERM2F128 $0x11, Y8, Y8, Y8 // (qw2, qw3, qw2, qw3)
	VPERM2F128 $0x00, Y1, Y1, Y9 // (c0, c1, c0, c1)
	VADDPS     Y8, Y9, Y10
	VSUBPS     Y8, Y9, Y11
	VBLENDPS   $0xF0, Y11, Y10, Y1
	VMOVUPS    Y1, (DI)(R10*8)
	ADDQ       $4, R10
	CMPQ       R10, SI
	JLT        fs12_loop

	MOVQ $4, R8                 // half = 4

half_loop:
	CMPQ R8, SI
	JGE  done
	LEAQ (DX)(R8*8), R9         // w = tw + half
	XORQ R10, R10               // i = 0

i_loop:
	LEAQ (DI)(R10*8), R11       // p = a + i
	LEAQ (R11)(R8*8), R12       // q = p + half
	MOVQ R9, R14                // wj = w
	MOVQ R8, R13                // j = half (counts down by 4)

j_loop:
	VMOVUPS   (R12), Y1         // q: 4 complexes
	VMOVUPS   (R14), Y2         // w: 4 twiddles
	VMOVSLDUP Y2, Y3            // wr lanes
	VMOVSHDUP Y2, Y4            // wi lanes
	VXORPS    Y15, Y4, Y4       // s*wi (exact sign flip)
	VPERMILPS $0xB1, Y1, Y5     // q with re/im swapped
	VMULPS    Y3, Y1, Y6        // t1 = q*wr
	VMULPS    Y4, Y5, Y7        // t2 = swap(q)*wi
	VADDSUBPS Y7, Y6, Y8        // v = (t1e-t2e, t1o+t2o)
	VMOVUPS   (R11), Y9         // p
	VADDPS    Y8, Y9, Y10       // p + v
	VSUBPS    Y8, Y9, Y11       // p - v
	VMOVUPS   Y10, (R11)
	VMOVUPS   Y11, (R12)
	ADDQ      $32, R11
	ADDQ      $32, R12
	ADDQ      $32, R14
	SUBQ      $4, R13
	JNZ       j_loop

	LEAQ (R10)(R8*2), R10       // i += half*2
	CMPQ R10, SI
	JLT  i_loop
	SHLQ $1, R8                 // half <<= 1
	JMP  half_loop

done:
	VZEROUPPER
	RET

// func fftColsBflySIMD(p, q []complex64, w complex64)
TEXT ·FFTColsBfly(SB), NOSPLIT, $0-56
	MOVQ         p_base+0(FP), DI
	MOVQ         p_len+8(FP), SI
	MOVQ         q_base+24(FP), DX
	VBROADCASTSS w_real+48(FP), Y3
	VBROADCASTSS w_imag+52(FP), Y4
	MOVQ         SI, CX
	ANDQ         $-4, CX
	XORQ         R10, R10
	CMPQ         CX, $0
	JEQ          tail

main_loop:
	VMOVUPS   (DX)(R10*8), Y1
	VPERMILPS $0xB1, Y1, Y5
	VMULPS    Y3, Y1, Y6
	VMULPS    Y4, Y5, Y7
	VADDSUBPS Y7, Y6, Y8
	VMOVUPS   (DI)(R10*8), Y9
	VADDPS    Y8, Y9, Y10
	VSUBPS    Y8, Y9, Y11
	VMOVUPS   Y10, (DI)(R10*8)
	VMOVUPS   Y11, (DX)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, CX
	JLT       main_loop

tail:
	CMPQ R10, SI
	JGE  done_cols

tail_loop:
	VMOVSD    (DX)(R10*8), X1   // one complex
	VPERMILPS $0xB1, X1, X5
	VMULPS    X3, X1, X6
	VMULPS    X4, X5, X7
	VADDSUBPS X7, X6, X8
	VMOVSD    (DI)(R10*8), X9
	VADDPS    X8, X9, X10
	VSUBPS    X8, X9, X11
	VMOVSD    X10, (DI)(R10*8)
	VMOVSD    X11, (DX)(R10*8)
	INCQ      R10
	CMPQ      R10, SI
	JLT       tail_loop

done_cols:
	VZEROUPPER
	RET

// func FFTCols4(r0, r1, r2, r3 []complex64, w1, w2a, w2b complex64)
// Two fused column-FFT stages on a closed row quad; each butterfly is the
// exact FFTColsBfly sequence, values held in registers between stages.
TEXT ·FFTCols4(SB), NOSPLIT, $0-120
	MOVQ         r0_base+0(FP), DI
	MOVQ         r0_len+8(FP), SI
	MOVQ         r1_base+24(FP), DX
	MOVQ         r2_base+48(FP), BX
	MOVQ         r3_base+72(FP), R8
	VBROADCASTSS w1_real+96(FP), Y0
	VBROADCASTSS w1_imag+100(FP), Y1
	VBROADCASTSS w2a_real+104(FP), Y2
	VBROADCASTSS w2a_imag+108(FP), Y3
	VBROADCASTSS w2b_real+112(FP), Y4
	VBROADCASTSS w2b_imag+116(FP), Y5
	MOVQ         SI, CX
	ANDQ         $-4, CX
	XORQ         R10, R10
	CMPQ         CX, $0
	JEQ          c4_tail

c4_main:
	VMOVUPS (DI)(R10*8), Y6     // A = r0
	VMOVUPS (DX)(R10*8), Y7     // B = r1
	VMOVUPS (BX)(R10*8), Y8     // C = r2
	VMOVUPS (R8)(R10*8), Y9     // D = r3

	// stage 1: A,B = A±B*w1 and C,D = C±D*w1
	VPERMILPS $0xB1, Y7, Y10
	VMULPS    Y0, Y7, Y11       // t1 = B*w1r
	VMULPS    Y1, Y10, Y10      // t2 = swap(B)*w1i
	VADDSUBPS Y10, Y11, Y11     // B*w1
	VADDPS    Y11, Y6, Y12      // A' = A + Bw
	VSUBPS    Y11, Y6, Y7       // B' = A - Bw
	VPERMILPS $0xB1, Y9, Y10
	VMULPS    Y0, Y9, Y11
	VMULPS    Y1, Y10, Y10
	VADDSUBPS Y10, Y11, Y11     // D*w1
	VADDPS    Y11, Y8, Y13      // C' = C + Dw
	VSUBPS    Y11, Y8, Y9       // D' = C - Dw

	// stage 2: A,C = A'±C'*w2a and B,D = B'±D'*w2b
	VPERMILPS $0xB1, Y13, Y10
	VMULPS    Y2, Y13, Y11
	VMULPS    Y3, Y10, Y10
	VADDSUBPS Y10, Y11, Y11     // C'*w2a
	VADDPS    Y11, Y12, Y6      // r0 out
	VSUBPS    Y11, Y12, Y8      // r2 out
	VPERMILPS $0xB1, Y9, Y10
	VMULPS    Y4, Y9, Y11
	VMULPS    Y5, Y10, Y10
	VADDSUBPS Y10, Y11, Y11     // D'*w2b
	VADDPS    Y11, Y7, Y12      // r1 out
	VSUBPS    Y11, Y7, Y9       // r3 out

	VMOVUPS Y6, (DI)(R10*8)
	VMOVUPS Y12, (DX)(R10*8)
	VMOVUPS Y8, (BX)(R10*8)
	VMOVUPS Y9, (R8)(R10*8)
	ADDQ    $4, R10
	CMPQ    R10, CX
	JLT     c4_main

c4_tail:
	CMPQ R10, SI
	JGE  c4_done

c4_tail_loop:
	VMOVSD    (DI)(R10*8), X6
	VMOVSD    (DX)(R10*8), X7
	VMOVSD    (BX)(R10*8), X8
	VMOVSD    (R8)(R10*8), X9
	VPERMILPS $0xB1, X7, X10
	VMULPS    X0, X7, X11
	VMULPS    X1, X10, X10
	VADDSUBPS X10, X11, X11
	VADDPS    X11, X6, X12
	VSUBPS    X11, X6, X7
	VPERMILPS $0xB1, X9, X10
	VMULPS    X0, X9, X11
	VMULPS    X1, X10, X10
	VADDSUBPS X10, X11, X11
	VADDPS    X11, X8, X13
	VSUBPS    X11, X8, X9
	VPERMILPS $0xB1, X13, X10
	VMULPS    X2, X13, X11
	VMULPS    X3, X10, X10
	VADDSUBPS X10, X11, X11
	VADDPS    X11, X12, X6
	VSUBPS    X11, X12, X8
	VPERMILPS $0xB1, X9, X10
	VMULPS    X4, X9, X11
	VMULPS    X5, X10, X10
	VADDSUBPS X10, X11, X11
	VADDPS    X11, X7, X12
	VSUBPS    X11, X7, X9
	VMOVSD    X6, (DI)(R10*8)
	VMOVSD    X12, (DX)(R10*8)
	VMOVSD    X8, (BX)(R10*8)
	VMOVSD    X9, (R8)(R10*8)
	INCQ      R10
	CMPQ      R10, SI
	JLT       c4_tail_loop

c4_done:
	VZEROUPPER
	RET

// func mulConjSIMD(spec, tspec []complex64)
// spec[i] = (ar*br + ai*bi) + (ai*br - ar*bi)i, via t1 = a*br,
// t2 = swap(a)*bi, addsub(t1, -t2) = (t1e+t2e, t1o-t2o); negation by xor
// and x-(-y) == x+y are IEEE-exact.
TEXT ·MulConj(SB), NOSPLIT, $0-48
	MOVQ spec_base+0(FP), DI
	MOVQ spec_len+8(FP), SI
	MOVQ tspec_base+24(FP), DX
	MOVL $0x80000000, AX
	MOVD AX, X0
	VPBROADCASTD X0, Y15
	MOVQ SI, CX
	ANDQ $-4, CX
	XORQ R10, R10
	CMPQ CX, $0
	JEQ  mc_tail

mc_main:
	VMOVUPS   (DI)(R10*8), Y1   // a
	VMOVUPS   (DX)(R10*8), Y2   // b
	VMOVSLDUP Y2, Y3            // br
	VMOVSHDUP Y2, Y4            // bi
	VPERMILPS $0xB1, Y1, Y5     // swap(a)
	VMULPS    Y3, Y1, Y6        // t1
	VMULPS    Y4, Y5, Y7        // t2
	VXORPS    Y15, Y7, Y7       // -t2
	VADDSUBPS Y7, Y6, Y8        // (t1e+t2e, t1o-t2o)
	VMOVUPS   Y8, (DI)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, CX
	JLT       mc_main

mc_tail:
	CMPQ R10, SI
	JGE  mc_done

mc_tail_loop:
	VMOVSD    (DI)(R10*8), X1
	VMOVSD    (DX)(R10*8), X2
	VMOVSLDUP X2, X3
	VMOVSHDUP X2, X4
	VPERMILPS $0xB1, X1, X5
	VMULPS    X3, X1, X6
	VMULPS    X4, X5, X7
	VXORPS    X15, X7, X7
	VADDSUBPS X7, X6, X8
	VMOVSD    X8, (DI)(R10*8)
	INCQ      R10
	CMPQ      R10, SI
	JLT       mc_tail_loop

mc_done:
	VZEROUPPER
	RET

DATA absmask<>+0(SB)/8, $0x7fffffffffffffff
DATA absmask<>+8(SB)/8, $0x7fffffffffffffff
DATA absmask<>+16(SB)/8, $0x7fffffffffffffff
DATA absmask<>+24(SB)/8, $0x7fffffffffffffff
GLOBL absmask<>(SB), RODATA|NOPTR, $32

DATA signmask<>+0(SB)/8, $0x8000000000000000
DATA signmask<>+8(SB)/8, $0x8000000000000000
DATA signmask<>+16(SB)/8, $0x8000000000000000
DATA signmask<>+24(SB)/8, $0x8000000000000000
GLOBL signmask<>(SB), RODATA|NOPTR, $32

DATA normconst<>+0(SB)/8, $0.5
DATA normconst<>+8(SB)/8, $1.0
DATA normconst<>+16(SB)/8, $1.125
GLOBL normconst<>(SB), RODATA|NOPTR, $24

// func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
//              mean *[4]float64, invArea, eps, templNorm float64)
// Register map: constants Y9..Y15 = 0.5, 1.0, 1.125, invArea, eps,
// templNorm, m0; m1..m3 live in Y6..Y8 when cn needs them.
TEXT ·NormRow(SB), NOSPLIT, $0-112
	MOVQ rrow_base+0(FP), DI
	MOVQ crow_base+24(FP), SI
	MOVQ wt+48(FP), R8          // t0
	MOVQ stride+56(FP), R9
	SHLQ $3, R9                 // stride in bytes
	MOVQ n+64(FP), DX
	MOVQ cn+72(FP), CX
	MOVQ mean+80(FP), BX

	VBROADCASTSD normconst<>+0(SB), Y9    // 0.5
	VBROADCASTSD normconst<>+8(SB), Y10   // 1.0
	VBROADCASTSD normconst<>+16(SB), Y11  // 1.125
	VBROADCASTSD invArea+88(FP), Y12
	VBROADCASTSD eps+96(FP), Y13
	VBROADCASTSD templNorm+104(FP), Y14
	VBROADCASTSD 0(BX), Y15               // m0

	LEAQ (R8)(R9*1), R10        // t1 (or q2 when cn==1)
	LEAQ (R10)(R9*1), R11       // t2
	LEAQ (R11)(R9*1), R12       // t3 (or q2 when cn==3)
	XORQ AX, AX                 // element index

	CMPQ CX, $3
	JEQ  cn3_setup
	CMPQ CX, $1
	JEQ  cn1_loop
	// cn == 4
	VBROADCASTSD 8(BX), Y6      // m1
	VBROADCASTSD 16(BX), Y7     // m2
	VBROADCASTSD 24(BX), Y8     // m3
	LEAQ (R12)(R9*1), R13       // q2 = wt + 4*stride

cn4_loop:
	CMPQ AX, DX
	JGE  norm_done
	VCVTPS2PD (SI)(AX*4), Y0    // num = crow
	VMOVUPD   (R8)(AX*8), Y1    // t0
	VMULPD    Y15, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y3        // wm2
	VMOVUPD   (R10)(AX*8), Y1   // t1
	VMULPD    Y6, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y2
	VADDPD    Y2, Y3, Y3
	VMOVUPD   (R11)(AX*8), Y1   // t2
	VMULPD    Y7, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y2
	VADDPD    Y2, Y3, Y3
	VMOVUPD   (R12)(AX*8), Y1   // t3
	VMULPD    Y8, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y2
	VADDPD    Y2, Y3, Y3
	VMOVUPD   (R13)(AX*8), Y4   // s2d
	CALL      norm_tail4<>(SB)
	ADDQ      $4, AX
	JMP       cn4_loop

cn3_setup:
	VBROADCASTSD 8(BX), Y6      // m1
	VBROADCASTSD 16(BX), Y7     // m2

cn3_loop:
	CMPQ AX, DX
	JGE  norm_done
	VCVTPS2PD (SI)(AX*4), Y0
	VMOVUPD   (R8)(AX*8), Y1
	VMULPD    Y15, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y3
	VMOVUPD   (R10)(AX*8), Y1
	VMULPD    Y6, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y2
	VADDPD    Y2, Y3, Y3
	VMOVUPD   (R11)(AX*8), Y1
	VMULPD    Y7, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y2
	VADDPD    Y2, Y3, Y3
	VMOVUPD   (R12)(AX*8), Y4   // q2 = wt + 3*stride
	CALL      norm_tail4<>(SB)
	ADDQ      $4, AX
	JMP       cn3_loop

cn1_loop:
	CMPQ AX, DX
	JGE  norm_done
	VCVTPS2PD (SI)(AX*4), Y0
	VMOVUPD   (R8)(AX*8), Y1
	VMULPD    Y15, Y1, Y2
	VSUBPD    Y2, Y0, Y0
	VMULPD    Y1, Y1, Y3
	VMOVUPD   (R10)(AX*8), Y4   // q2 = wt + stride
	CALL      norm_tail4<>(SB)
	ADDQ      $4, AX
	JMP       cn1_loop

norm_done:
	VZEROUPPER
	RET

// norm_tail4: shared per-vector tail. In: Y0 = num, Y3 = wm2 (pre
// invArea), Y4 = s2d; constants Y9=0.5 Y10=1.0 Y11=1.125 Y12=invArea
// Y13=eps Y14=templNorm; AX = element index, DI = rrow. Clobbers Y0..Y5.
TEXT norm_tail4<>(SB), NOSPLIT, $0-0
	VMULPD  Y12, Y3, Y3         // wm2 *= invArea
	VSUBPD  Y3, Y4, Y1          // diff2 = s2d - wm2
	VXORPD  Y2, Y2, Y2
	VMAXPD  Y2, Y1, Y1          // clamp at 0 (diff2 is never -0/NaN)
	VMULPD  Y13, Y4, Y5         // e = eps*s2d
	VMINPD  Y5, Y9, Y5          // lim = 0.5 < e ? 0.5 : e
	VSQRTPD Y1, Y2
	VMULPD  Y14, Y2, Y2         // sq = sqrt(diff2)*templNorm
	VCMPPD  $0x12, Y5, Y1, Y3   // diff2 <= lim (LE_OQ)
	VANDNPD Y2, Y3, Y3          // den = mask ? 0 : sq
	VANDPD  absmask<>(SB), Y0, Y5 // an = |num|
	VCMPPD  $0x11, Y3, Y5, Y4   // m1 = an < den (LT_OQ)
	VANDPD  Y4, Y0, Y1          // numer = m1 ? num : 0
	VBLENDVPD Y4, Y3, Y10, Y2   // divisor = m1 ? den : 1.0
	VDIVPD  Y2, Y1, Y1          // dv
	VMULPD  Y11, Y3, Y2         // den*1.125
	VCMPPD  $0x11, Y2, Y5, Y5   // m2 = an < den*1.125
	VANDPD  signmask<>(SB), Y0, Y2
	VORPD   Y10, Y2, Y2         // sat = copysign(1, num); only selected
	                            // when num is nonzero, where it equals
	                            // the scalar num>0 ? 1 : -1
	VANDPD  Y5, Y2, Y2          // inner = m2 ? sat : 0
	VBLENDVPD Y4, Y1, Y2, Y2    // v = m1 ? dv : inner
	VCVTPD2PSY Y2, X2
	VMOVUPS X2, (DI)(AX*4)
	RET

// func PackRows2(z []complex64, ra, rb []uint8, step int)
// z[i] = (float32(ra[i*step]), float32(rb[i*step])). Exact conversions,
// pure data movement. step is 1 or 4.
TEXT ·PackRows2(SB), NOSPLIT, $0-80
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ ra_base+24(FP), SI
	MOVQ rb_base+48(FP), BX
	MOVQ step+72(FP), CX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11
	CMPQ CX, $4
	JEQ  pk2_clamp4
	JMP  pk2_s1

pk2_clamp4:
	// Each stride-4 iteration gathers with a full 16-byte load, which
	// reads 3 bytes past the last used byte. Never read past the row
	// slice (the image may end exactly at a page boundary): allow only
	// floor(len/16) vector iterations per row, tail handles the rest.
	MOVQ ra_len+32(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11
	MOVQ rb_len+56(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11

pk2_s4:
	CMPQ R10, R11
	JGE  pk2_tail
	// gather 4 bytes at stride 4 from each row via PSHUFB
	VMOVDQU    (SI), X1
	VMOVDQU    (BX), X2
	VPSHUFB    pkshuf4<>(SB), X1, X1  // bytes 0,4,8,12 -> lanes 0..3
	VPSHUFB    pkshuf4<>(SB), X2, X2
	VPMOVZXBD  X1, X1
	VPMOVZXBD  X2, X2
	VCVTDQ2PS  X1, X1                 // 4 floats (re)
	VCVTDQ2PS  X2, X2                 // 4 floats (im)
	VUNPCKLPS  X2, X1, X3             // re0 im0 re1 im1
	VUNPCKHPS  X2, X1, X4             // re2 im2 re3 im3
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $16, SI
	ADDQ       $16, BX
	ADDQ       $4, R10
	JMP        pk2_s4

pk2_s1:
	CMPQ R10, R11
	JGE  pk2_tail
	VPMOVZXBD  (SI), X1               // 4 bytes -> 4 dwords
	VPMOVZXBD  (BX), X2
	VCVTDQ2PS  X1, X1
	VCVTDQ2PS  X2, X2
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $4, SI
	ADDQ       $4, BX
	ADDQ       $4, R10
	JMP        pk2_s1

pk2_tail:
	CMPQ R10, DX
	JGE  pk2_done
	MOVBLZX (SI), AX
	VCVTSI2SSL AX, X1, X1
	MOVBLZX (BX), AX
	VCVTSI2SSL AX, X2, X2
	VMOVSS  X1, (DI)(R10*8)
	VMOVSS  X2, 4(DI)(R10*8)
	ADDQ    CX, SI
	ADDQ    CX, BX
	INCQ    R10
	JMP     pk2_tail

pk2_done:
	VZEROUPPER
	RET

// func PackRows1(z []complex64, ra []uint8, step int)
TEXT ·PackRows1(SB), NOSPLIT, $0-56
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ ra_base+24(FP), SI
	MOVQ step+48(FP), CX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11
	VXORPS X2, X2, X2
	CMPQ CX, $4
	JEQ  pk1_clamp4
	JMP  pk1_s1

pk1_clamp4:
	// Same in-bounds clamp as PackRows2 (16-byte gather per iteration).
	MOVQ ra_len+32(FP), AX
	SHRQ $4, AX
	SHLQ $2, AX
	CMPQ AX, R11
	CMOVQLT AX, R11

pk1_s4:
	CMPQ R10, R11
	JGE  pk1_tail
	VMOVDQU    (SI), X1
	VPSHUFB    pkshuf4<>(SB), X1, X1
	VPMOVZXBD  X1, X1
	VCVTDQ2PS  X1, X1
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $16, SI
	ADDQ       $4, R10
	JMP        pk1_s4

pk1_s1:
	CMPQ R10, R11
	JGE  pk1_tail
	VPMOVZXBD  (SI), X1
	VCVTDQ2PS  X1, X1
	VUNPCKLPS  X2, X1, X3
	VUNPCKHPS  X2, X1, X4
	VMOVUPS    X3, (DI)(R10*8)
	VMOVUPS    X4, 16(DI)(R10*8)
	ADDQ       $4, SI
	ADDQ       $4, R10
	JMP        pk1_s1

pk1_tail:
	CMPQ R10, DX
	JGE  pk1_done
	MOVBLZX (SI), AX
	VCVTSI2SSL AX, X1, X1
	VMOVSS  X1, (DI)(R10*8)
	MOVL    $0, 4(DI)(R10*8)
	ADDQ    CX, SI
	INCQ    R10
	JMP     pk1_tail

pk1_done:
	VZEROUPPER
	RET

DATA pkshuf4<>+0(SB)/8, $0x808080800c080400
DATA pkshuf4<>+8(SB)/8, $0x8080808080808080
GLOBL pkshuf4<>(SB), RODATA|NOPTR, $16

// func Untangle(sa, sb, z []complex64, n, k0, k1 int)
// sa[k] = 0.5*(zk.re+zn.re, zk.im-zn.im), sb[k] = 0.5*(zk.im+zn.im,
// zn.re-zk.re), zn = z[n-k]. k in [k0, k1), k0 >= 1. Negation by xor and
// x-(-y) == x+y are IEEE-exact, so every element gets the scalar op
// sequence: one rounded add/sub, one rounded multiply by 0.5.
TEXT ·Untangle(SB), NOSPLIT, $0-96
	MOVQ sa_base+0(FP), DI
	MOVQ sb_base+24(FP), SI
	MOVQ z_base+48(FP), DX
	MOVQ n+72(FP), R8
	MOVQ k0+80(FP), R10
	MOVQ k1+88(FP), R11
	MOVL $0x80000000, AX
	MOVD AX, X0
	VPBROADCASTD X0, Y15              // sign mask
	VBROADCASTSS half32<>(SB), Y14    // 0.5f
	MOVQ R11, R12
	SUBQ R10, R12
	ANDQ $-4, R12
	ADDQ R10, R12                     // vector end

ut_loop:
	CMPQ R10, R12
	JGE  ut_tail
	VMOVUPS   (DX)(R10*8), Y1         // zk: z[k..k+3]
	MOVQ      R8, R13
	SUBQ      R10, R13                // n-k
	VMOVUPS   -24(DX)(R13*8), Y2      // z[n-k-3..n-k]
	VPERMPD   $0x1B, Y2, Y2           // reverse -> zn for k..k+3
	VXORPS    Y15, Y2, Y3             // -zn
	VADDSUBPS Y3, Y1, Y4              // (zk.re+zn.re, zk.im-zn.im)
	VMULPS    Y14, Y4, Y4
	VMOVUPS   Y4, (DI)(R10*8)         // sa[k..k+3]
	VPERMILPS $0xB1, Y1, Y5           // zk swapped (im, re)
	VPERMILPS $0xB1, Y2, Y6           // zn swapped
	VXORPS    Y15, Y5, Y5             // -zk swapped
	VADDSUBPS Y5, Y6, Y7              // (zn.im+zk.im, zn.re-zk.re)
	VMULPS    Y14, Y7, Y7
	VMOVUPS   Y7, (SI)(R10*8)         // sb[k..k+3]
	ADDQ      $4, R10
	JMP       ut_loop

ut_tail:
	CMPQ R10, R11
	JGE  ut_done
	MOVQ      R8, R13
	SUBQ      R10, R13
	VMOVSD    (DX)(R10*8), X1         // zk
	VMOVSD    (DX)(R13*8), X2         // zn
	VXORPS    X15, X2, X3
	VADDSUBPS X3, X1, X4
	VMULPS    X14, X4, X4
	VMOVSD    X4, (DI)(R10*8)
	VPERMILPS $0xB1, X1, X5
	VPERMILPS $0xB1, X2, X6
	VXORPS    X15, X5, X5
	VADDSUBPS X5, X6, X7
	VMULPS    X14, X7, X7
	VMOVSD    X7, (SI)(R10*8)
	INCQ      R10
	JMP       ut_tail

ut_done:
	VZEROUPPER
	RET

DATA half32<>+0(SB)/4, $0.5
GLOBL half32<>(SB), RODATA|NOPTR, $4

// func CombineLow(z, sa, sb []complex64)
// z[k] = (sa.re - sb.im, sa.im + sb.re): exactly one addsub per element.
TEXT ·CombineLow(SB), NOSPLIT, $0-72
	MOVQ z_base+0(FP), DI
	MOVQ z_len+8(FP), DX
	MOVQ sa_base+24(FP), SI
	MOVQ sb_base+48(FP), BX
	XORQ R10, R10
	MOVQ DX, R11
	ANDQ $-4, R11

cl_loop:
	CMPQ R10, R11
	JGE  cl_tail
	VMOVUPS   (SI)(R10*8), Y1
	VMOVUPS   (BX)(R10*8), Y2
	VPERMILPS $0xB1, Y2, Y2           // (sb.im, sb.re)
	VADDSUBPS Y2, Y1, Y3              // (sa.re-sb.im, sa.im+sb.re)
	VMOVUPS   Y3, (DI)(R10*8)
	ADDQ      $4, R10
	JMP       cl_loop

cl_tail:
	CMPQ R10, DX
	JGE  cl_done
	VMOVSD    (SI)(R10*8), X1
	VMOVSD    (BX)(R10*8), X2
	VPERMILPS $0xB1, X2, X2
	VADDSUBPS X2, X1, X3
	VMOVSD    X3, (DI)(R10*8)
	INCQ      R10
	JMP       cl_tail

cl_done:
	VZEROUPPER
	RET

// func CombineHigh(z, sa, sb []complex64, n, hw int)
// z[k] = (sa[m].re + sb[m].im, sb[m].re - sa[m].im), m = n-k, k in [hw, n).
TEXT ·CombineHigh(SB), NOSPLIT, $0-88
	MOVQ z_base+0(FP), DI
	MOVQ sa_base+24(FP), SI
	MOVQ sb_base+48(FP), BX
	MOVQ n+72(FP), R8
	MOVQ hw+80(FP), R10
	MOVL $0x80000000, AX
	MOVD AX, X0
	VPBROADCASTD X0, Y15
	MOVQ R8, R12
	SUBQ R10, R12
	ANDQ $-4, R12
	ADDQ R10, R12                     // vector end

ch_loop:
	CMPQ R10, R12
	JGE  ch_tail
	MOVQ      R8, R13
	SUBQ      R10, R13                // m = n-k
	VMOVUPS   -24(SI)(R13*8), Y1      // sa[m-3..m]
	VMOVUPS   -24(BX)(R13*8), Y2      // sb[m-3..m]
	VPERMPD   $0x1B, Y1, Y1           // reverse -> sa[m] for k..k+3
	VPERMPD   $0x1B, Y2, Y2
	VPERMILPS $0xB1, Y2, Y2           // (sb.im, sb.re)
	VXORPS    Y15, Y1, Y1             // -sa
	VADDSUBPS Y1, Y2, Y3              // (sb.im+sa.re, sb.re-sa.im)
	VMOVUPS   Y3, (DI)(R10*8)
	ADDQ      $4, R10
	JMP       ch_loop

ch_tail:
	CMPQ R10, R8
	JGE  ch_done
	MOVQ      R8, R13
	SUBQ      R10, R13
	VMOVSD    (SI)(R13*8), X1
	VMOVSD    (BX)(R13*8), X2
	VPERMILPS $0xB1, X2, X2
	VXORPS    X15, X1, X1
	VADDSUBPS X1, X2, X3
	VMOVSD    X3, (DI)(R10*8)
	INCQ      R10
	JMP       ch_tail

ch_done:
	VZEROUPPER
	RET

// func MinMaxRow(row []float32) (minV, maxV float32, minI, maxI int)
// Tracks (value, index) pairs per lane with strict ordered compares, then
// reduces lanes by the lexicographic (value, index) order. Every compare
// is an exact predicate, so the result matches the scalar first-occurrence
// scan on any input (NaNs are never selected, as with scalar v<min).
// len(row) must be a nonzero multiple of 8, below 2^31.
TEXT ·MinMaxRow(SB), NOSPLIT, $0-48
	MOVQ    row_base+0(FP), DI
	MOVQ    row_len+8(FP), SI
	VMOVUPS (DI), Y0            // minV = row[0..7]
	VMOVUPS (DI), Y1            // maxV
	VMOVDQU idx8<>(SB), Y2      // minIdx = 0..7
	VMOVDQU idx8<>(SB), Y3      // maxIdx
	VMOVDQU idx8<>(SB), Y4      // curIdx
	VPBROADCASTD stride8<>(SB), Y5
	MOVQ    $8, R10

mm_loop:
	CMPQ      R10, SI
	JGE       mm_reduce
	VMOVUPS   (DI)(R10*4), Y6   // v
	VPADDD    Y5, Y4, Y4        // curIdx += 8
	VCMPPS    $0x11, Y0, Y6, Y7 // v < minV (LT_OQ)
	VBLENDVPS Y7, Y6, Y0, Y0
	VBLENDVPS Y7, Y4, Y2, Y2
	VCMPPS    $0x1E, Y1, Y6, Y7 // v > maxV (GT_OQ)
	VBLENDVPS Y7, Y6, Y1, Y1
	VBLENDVPS Y7, Y4, Y3, Y3
	ADDQ      $8, R10
	JMP       mm_loop

	// Lane tournament: fold 8 (value, index) lanes to lane 0 in three
	// swap/compare rounds; take = (v'<v) | (v'==v & i'<i).
mm_reduce:
	VPERM2F128 $0x01, Y0, Y0, Y6
	VPERM2F128 $0x01, Y2, Y2, Y7
	VCMPPS     $0x11, Y0, Y6, Y8
	VCMPPS     $0x00, Y0, Y6, Y9
	VPCMPGTD   Y7, Y2, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y0, Y0
	VBLENDVPS  Y8, Y7, Y2, Y2
	VPERMILPS  $0x4E, Y0, Y6
	VPERMILPS  $0x4E, Y2, Y7
	VCMPPS     $0x11, Y0, Y6, Y8
	VCMPPS     $0x00, Y0, Y6, Y9
	VPCMPGTD   Y7, Y2, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y0, Y0
	VBLENDVPS  Y8, Y7, Y2, Y2
	VPERMILPS  $0xB1, Y0, Y6
	VPERMILPS  $0xB1, Y2, Y7
	VCMPPS     $0x11, Y0, Y6, Y8
	VCMPPS     $0x00, Y0, Y6, Y9
	VPCMPGTD   Y7, Y2, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y0, Y0
	VBLENDVPS  Y8, Y7, Y2, Y2

	VPERM2F128 $0x01, Y1, Y1, Y6
	VPERM2F128 $0x01, Y3, Y3, Y7
	VCMPPS     $0x1E, Y1, Y6, Y8
	VCMPPS     $0x00, Y1, Y6, Y9
	VPCMPGTD   Y7, Y3, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y1, Y1
	VBLENDVPS  Y8, Y7, Y3, Y3
	VPERMILPS  $0x4E, Y1, Y6
	VPERMILPS  $0x4E, Y3, Y7
	VCMPPS     $0x1E, Y1, Y6, Y8
	VCMPPS     $0x00, Y1, Y6, Y9
	VPCMPGTD   Y7, Y3, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y1, Y1
	VBLENDVPS  Y8, Y7, Y3, Y3
	VPERMILPS  $0xB1, Y1, Y6
	VPERMILPS  $0xB1, Y3, Y7
	VCMPPS     $0x1E, Y1, Y6, Y8
	VCMPPS     $0x00, Y1, Y6, Y9
	VPCMPGTD   Y7, Y3, Y10
	VPAND      Y10, Y9, Y9
	VPOR       Y9, Y8, Y8
	VBLENDVPS  Y8, Y6, Y1, Y1
	VBLENDVPS  Y8, Y7, Y3, Y3

	VMOVSS X0, minV+24(FP)
	VMOVSS X1, maxV+28(FP)
	VMOVD  X2, AX
	MOVQ   AX, minI+32(FP)
	VMOVD  X3, AX
	MOVQ   AX, maxI+40(FP)
	VZEROUPPER
	RET

DATA idx8<>+0(SB)/4, $0
DATA idx8<>+4(SB)/4, $1
DATA idx8<>+8(SB)/4, $2
DATA idx8<>+12(SB)/4, $3
DATA idx8<>+16(SB)/4, $4
DATA idx8<>+20(SB)/4, $5
DATA idx8<>+24(SB)/4, $6
DATA idx8<>+28(SB)/4, $7
GLOBL idx8<>(SB), RODATA|NOPTR, $32

DATA stride8<>+0(SB)/4, $8
GLOBL stride8<>(SB), RODATA|NOPTR, $4

// func RGBAToGray(dst, src []uint8)
// gray = (4899*R + 9617*G + 1868*B + 8192) >> 14 — OpenCV's fixed-point
// BT.601 weights. Pure integer arithmetic (VPMADDWD sums stay far below
// 2^31), bit-identical to the scalar expression. Processes len(dst)
// pixels, a multiple of 8; reads exactly 4*len(dst) bytes of src.
TEXT ·RGBAToGray(SB), NOSPLIT, $0-48
	MOVQ         dst_base+0(FP), DI
	MOVQ         dst_len+8(FP), DX
	MOVQ         src_base+24(FP), SI
	VMOVDQU      graycoef<>(SB), Y12
	VPBROADCASTD grayround<>(SB), Y13
	VMOVDQU      graygather<>(SB), Y14
	XORQ         R10, R10
	CMPQ         DX, $0
	JE           g2y_done

g2y_loop:
	VPMOVZXBW    (SI), Y1       // pixels 0-3 as 16 words (r,g,b,a)
	VPMOVZXBW    16(SI), Y2     // pixels 4-7
	VPMADDWD     Y12, Y1, Y1    // (4899r+9617g, 1868b+0) per pixel
	VPMADDWD     Y12, Y2, Y2
	VPHADDD      Y2, Y1, Y1     // [P0,P1,P4,P5 | P2,P3,P6,P7]
	VPADDD       Y13, Y1, Y1    // + 8192
	VPSRLD       $14, Y1, Y1
	VPSHUFB      Y14, Y1, Y1    // gather low bytes into position
	VEXTRACTI128 $1, Y1, X2
	VPOR         X2, X1, X1     // 8 gray bytes
	VMOVQ        X1, (DI)(R10*1)
	ADDQ         $32, SI
	ADDQ         $8, R10
	CMPQ         R10, DX
	JLT          g2y_loop

g2y_done:
	VZEROUPPER
	RET

DATA graycoef<>+0(SB)/8, $0x0000074C25911323
DATA graycoef<>+8(SB)/8, $0x0000074C25911323
DATA graycoef<>+16(SB)/8, $0x0000074C25911323
DATA graycoef<>+24(SB)/8, $0x0000074C25911323
GLOBL graycoef<>(SB), RODATA|NOPTR, $32

DATA grayround<>+0(SB)/4, $8192
GLOBL grayround<>(SB), RODATA|NOPTR, $4

DATA graygather<>+0(SB)/8, $0x80800C0880800400
DATA graygather<>+8(SB)/8, $0x8080808080808080
DATA graygather<>+16(SB)/8, $0x0C08808004008080
DATA graygather<>+24(SB)/8, $0x8080808080808080
GLOBL graygather<>(SB), RODATA|NOPTR, $32

// func SlideCols1(colSum []int32, colSum2 []int64, rsub, radd []uint8)
// colSum[x] += a-b; colSum2[x] += a²-b² (a=radd[x], b=rsub[x]). All
// integer, exact at every width used (bytes, squares <= 65025, int64
// accumulate), so identical to the scalar loop.
TEXT ·SlideCols1(SB), NOSPLIT, $0-96
	MOVQ colSum_base+0(FP), DI
	MOVQ colSum_len+8(FP), CX
	MOVQ colSum2_base+24(FP), DX
	MOVQ rsub_base+48(FP), SI
	MOVQ radd_base+72(FP), BX
	XORQ R10, R10

sc1_loop:
	CMPQ         R10, CX
	JGE          sc1_done
	VPMOVZXBD    (BX)(R10*1), Y1   // a
	VPMOVZXBD    (SI)(R10*1), Y2   // b
	VPSUBD       Y2, Y1, Y3
	VPADDD       (DI)(R10*4), Y3, Y3
	VMOVDQU      Y3, (DI)(R10*4)
	VPMULLD      Y1, Y1, Y4        // a²
	VPMULLD      Y2, Y2, Y5        // b²
	VPSUBD       Y5, Y4, Y4
	VPMOVSXDQ    X4, Y5            // low 4 diffs → int64
	VEXTRACTI128 $1, Y4, X6
	VPMOVSXDQ    X6, Y6
	VPADDQ       (DX)(R10*8), Y5, Y5
	VMOVDQU      Y5, (DX)(R10*8)
	VPADDQ       32(DX)(R10*8), Y6, Y6
	VMOVDQU      Y6, 32(DX)(R10*8)
	ADDQ         $8, R10
	JMP          sc1_loop

sc1_done:
	VZEROUPPER
	RET

// func SlideCols4(colSum []int32, colSum2 []int64, rsub, radd []uint8, cn int)
// RGBA layout: 4 colSum lanes per pixel; colSum2 gets the summed squared
// channel deltas, alpha words masked to zero when cn == 3. VPMADDWD pairs
// (r²+g², b²+a²) stay far below 2^31; integer reassociation is exact.
TEXT ·SlideCols4(SB), NOSPLIT, $0-104
	MOVQ     colSum_base+0(FP), DI
	MOVQ     colSum2_base+24(FP), DX
	MOVQ     colSum2_len+32(FP), CX  // pixels, multiple of 8
	MOVQ     rsub_base+48(FP), SI
	MOVQ     radd_base+72(FP), BX
	MOVQ     cn+96(FP), AX
	VMOVDQU  permlin<>(SB), Y13
	VPCMPEQD Y14, Y14, Y14           // cn=4: mask is a no-op
	CMPQ     AX, $3
	JNE      sc4_start
	VMOVDQU  alphamask<>(SB), Y14    // zero the alpha words

sc4_start:
	TESTQ CX, CX
	JZ    sc4_done

sc4_loop:
	// colSum += a-b over 32 int32 lanes (8 RGBA pixels)
	VPMOVZXBD (BX), Y1
	VPMOVZXBD (SI), Y2
	VPSUBD    Y2, Y1, Y3
	VPADDD    (DI), Y3, Y3
	VMOVDQU   Y3, (DI)
	VPMOVZXBD 8(BX), Y1
	VPMOVZXBD 8(SI), Y2
	VPSUBD    Y2, Y1, Y3
	VPADDD    32(DI), Y3, Y3
	VMOVDQU   Y3, 32(DI)
	VPMOVZXBD 16(BX), Y1
	VPMOVZXBD 16(SI), Y2
	VPSUBD    Y2, Y1, Y3
	VPADDD    64(DI), Y3, Y3
	VMOVDQU   Y3, 64(DI)
	VPMOVZXBD 24(BX), Y1
	VPMOVZXBD 24(SI), Y2
	VPSUBD    Y2, Y1, Y3
	VPADDD    96(DI), Y3, Y3
	VMOVDQU   Y3, 96(DI)

	// colSum2 += sum of squared channel deltas per pixel
	VPMOVZXBW    (BX), Y4            // pixels 0-3 as words
	VPMOVZXBW    (SI), Y5
	VPAND        Y14, Y4, Y4
	VPAND        Y14, Y5, Y5
	VPMADDWD     Y4, Y4, Y4          // (r²+g², b²+a²) per pixel
	VPMADDWD     Y5, Y5, Y5
	VPSUBD       Y5, Y4, Y4
	VPMOVZXBW    16(BX), Y6          // pixels 4-7
	VPMOVZXBW    16(SI), Y7
	VPAND        Y14, Y6, Y6
	VPAND        Y14, Y7, Y7
	VPMADDWD     Y6, Y6, Y6
	VPMADDWD     Y7, Y7, Y7
	VPSUBD       Y7, Y6, Y6
	VPHADDD      Y6, Y4, Y4          // [q0,q1,q4,q5 | q2,q3,q6,q7]
	VPERMD       Y4, Y13, Y4         // [q0..q7]
	VPMOVSXDQ    X4, Y5
	VEXTRACTI128 $1, Y4, X6
	VPMOVSXDQ    X6, Y6
	VPADDQ       (DX), Y5, Y5
	VMOVDQU      Y5, (DX)
	VPADDQ       32(DX), Y6, Y6
	VMOVDQU      Y6, 32(DX)

	ADDQ $32, BX
	ADDQ $32, SI
	ADDQ $128, DI
	ADDQ $64, DX
	SUBQ $8, CX
	JNZ  sc4_loop

sc4_done:
	VZEROUPPER
	RET

DATA alphamask<>+0(SB)/8, $0x0000FFFFFFFFFFFF
DATA alphamask<>+8(SB)/8, $0x0000FFFFFFFFFFFF
DATA alphamask<>+16(SB)/8, $0x0000FFFFFFFFFFFF
DATA alphamask<>+24(SB)/8, $0x0000FFFFFFFFFFFF
GLOBL alphamask<>(SB), RODATA|NOPTR, $32

DATA permlin<>+0(SB)/4, $0
DATA permlin<>+4(SB)/4, $1
DATA permlin<>+8(SB)/4, $4
DATA permlin<>+12(SB)/4, $5
DATA permlin<>+16(SB)/4, $2
DATA permlin<>+20(SB)/4, $3
DATA permlin<>+24(SB)/4, $6
DATA permlin<>+28(SB)/4, $7
GLOBL permlin<>(SB), RODATA|NOPTR, $32

// func SlideSpill1(wt, q2 []float64, lo, hi []int32, lo2, hi2 []int64,
//	s0, s2 int64) (ns0, ns2 int64)
// The cn=1 normalize spill: wt[i]=float64(s0), q2[i]=float64(s2), then
// s0 += hi[i]-lo[i], s2 += hi2[i]-lo2[i]. The sums are exact nonnegative
// integers < 2^52, so float64(v) == asDouble(v | 2^52bits) - 2^52 with no
// rounding anywhere; the integer chains run in scalar registers exactly
// like the Go loop.
TEXT ·SlideSpill1(SB), NOSPLIT, $0-176
	MOVQ wt_base+0(FP), DI
	MOVQ wt_len+8(FP), DX
	MOVQ q2_base+24(FP), SI
	MOVQ lo_base+48(FP), R8
	MOVQ hi_base+72(FP), R9
	MOVQ lo2_base+96(FP), R10
	MOVQ hi2_base+120(FP), R11
	MOVQ s0+144(FP), AX
	MOVQ s2+152(FP), BX
	MOVQ $0x4330000000000000, R12 // 2^52: OR-mask and, as a double, the bias
	MOVQ R12, X14
	PUNPCKLQDQ X14, X14
	XORQ CX, CX
	MOVQ DX, R15
	ANDQ $-2, R15

ss1_loop:
	CMPQ CX, R15
	JGE  ss1_tail
	MOVLQSX (R9)(CX*4), R12     // hi[i]
	MOVLQSX (R8)(CX*4), R13     // lo[i]
	SUBQ    R13, R12
	LEAQ    (AX)(R12*1), R13    // s1 = s0 + d0
	MOVQ    AX, X0
	MOVQ    R13, X1
	PUNPCKLQDQ X1, X0           // (s0, s1)
	POR     X14, X0
	SUBPD   X14, X0             // exact int64 -> float64 pair
	MOVUPS  X0, (DI)(CX*8)
	MOVLQSX 4(R9)(CX*4), R12
	MOVLQSX 4(R8)(CX*4), AX     // s0's old value is dead; reuse as scratch
	SUBQ    AX, R12
	LEAQ    (R13)(R12*1), AX    // s0 advanced past both
	MOVQ    (R11)(CX*8), R12    // hi2[i]
	SUBQ    (R10)(CX*8), R12
	LEAQ    (BX)(R12*1), R13    // t1 = s2 + e0
	MOVQ    BX, X2
	MOVQ    R13, X3
	PUNPCKLQDQ X3, X2
	POR     X14, X2
	SUBPD   X14, X2
	MOVUPS  X2, (SI)(CX*8)
	MOVQ    8(R11)(CX*8), R12
	SUBQ    8(R10)(CX*8), R12
	LEAQ    (R13)(R12*1), BX    // s2 advanced
	ADDQ    $2, CX
	JMP     ss1_loop

ss1_tail:
	CMPQ CX, DX
	JGE  ss1_done
	CVTSQ2SD AX, X0             // exact below 2^52
	MOVSD    X0, (DI)(CX*8)
	CVTSQ2SD BX, X1
	MOVSD    X1, (SI)(CX*8)
	MOVLQSX  (R9)(CX*4), R12
	MOVLQSX  (R8)(CX*4), R13
	SUBQ     R13, R12
	ADDQ     R12, AX
	MOVQ     (R11)(CX*8), R12
	SUBQ     (R10)(CX*8), R12
	ADDQ     R12, BX
	INCQ     CX
	JMP      ss1_tail

ss1_done:
	MOVQ AX, ns0+160(FP)
	MOVQ BX, ns2+168(FP)
	RET

// func EmitRe(dst []float32, z []complex64, add bool)
TEXT ·EmitRe(SB), NOSPLIT, $0-49
	MOVQ    dst_base+0(FP), DI
	MOVQ    dst_len+8(FP), DX
	MOVQ    z_base+24(FP), SI
	MOVBLZX add+48(FP), CX
	XORQ    R10, R10
	MOVQ    DX, R11
	ANDQ    $-8, R11

er_loop:
	CMPQ R10, R11
	JGE  er_tail
	VMOVUPS  (SI)(R10*8), Y1
	VMOVUPS  32(SI)(R10*8), Y2
	VSHUFPS  $0x88, Y2, Y1, Y3        // re lanes, 128-bit interleaved
	VPERMPD  $0xD8, Y3, Y3            // fix lane order
	TESTQ    CX, CX
	JZ       er_store
	VADDPS   (DI)(R10*4), Y3, Y3

er_store:
	VMOVUPS  Y3, (DI)(R10*4)
	ADDQ     $8, R10
	JMP      er_loop

er_tail:
	CMPQ R10, DX
	JGE  er_done
	VMOVSS (SI)(R10*8), X1
	TESTQ  CX, CX
	JZ     er_tstore
	VADDSS (DI)(R10*4), X1, X1

er_tstore:
	VMOVSS X1, (DI)(R10*4)
	INCQ   R10
	JMP    er_tail

er_done:
	VZEROUPPER
	RET

// func EmitIm(dst []float32, z []complex64, add bool)
TEXT ·EmitIm(SB), NOSPLIT, $0-49
	MOVQ    dst_base+0(FP), DI
	MOVQ    dst_len+8(FP), DX
	MOVQ    z_base+24(FP), SI
	MOVBLZX add+48(FP), CX
	XORQ    R10, R10
	MOVQ    DX, R11
	ANDQ    $-8, R11

ei_loop:
	CMPQ R10, R11
	JGE  ei_tail
	VMOVUPS  (SI)(R10*8), Y1
	VMOVUPS  32(SI)(R10*8), Y2
	VSHUFPS  $0xDD, Y2, Y1, Y3        // im lanes
	VPERMPD  $0xD8, Y3, Y3
	TESTQ    CX, CX
	JZ       ei_store
	VADDPS   (DI)(R10*4), Y3, Y3

ei_store:
	VMOVUPS  Y3, (DI)(R10*4)
	ADDQ     $8, R10
	JMP      ei_loop

ei_tail:
	CMPQ R10, DX
	JGE  ei_done
	VMOVSS 4(SI)(R10*8), X1
	TESTQ  CX, CX
	JZ     ei_tstore
	VADDSS (DI)(R10*4), X1, X1

ei_tstore:
	VMOVSS X1, (DI)(R10*4)
	INCQ   R10
	JMP    ei_tail

ei_done:
	VZEROUPPER
	RET
