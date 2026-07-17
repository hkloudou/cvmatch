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
