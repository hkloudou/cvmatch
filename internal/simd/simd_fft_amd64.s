//go:build gc && !purego

#include "textflag.h"
// FFT kernels: the radix-2 stage cascade, column-direction butterflies
// (single stage and fused stage pairs), and the conjugate spectrum
// multiply.

// Part of the AVX2 kernel set — bit-identical to the generic Go loops.
// Shared conventions (see simd_amd64.s for the CPU-detection entry):
// complex64 values are interleaved (re, im) float32 pairs; the butterfly
// evaluates t1 = q*wr, t2 = swap(q)*wi and addsub(t1, t2) =
// (t1e-t2e, t1o+t2o), which reproduces the scalar complex multiply's two
// products and single rounded add/sub per component (the imaginary sum
// is commuted; IEEE addition commutes bit-exactly). Never VFMADD*: every
// multiply and add rounds separately, exactly like the scalar code.

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
