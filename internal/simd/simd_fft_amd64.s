//go:build gc && !purego

#include "textflag.h"
// FFT kernels: the radix-4 row cascade, radix-4 column stages with
// their head pass, and the conjugate spectrum multiply.

// Part of the AVX2 kernel set — bit-identical to the generic Go loops.
// Shared conventions (see simd_amd64.s for the CPU-detection entry):
// complex64 values are interleaved (re, im) float32 pairs; the butterfly
// evaluates t1 = q*wr, t2 = swap(q)*wi and addsub(t1, t2) =
// (t1e-t2e, t1o+t2o), which reproduces the scalar complex multiply's two
// products and single rounded add/sub per component (the imaginary sum
// is commuted; IEEE addition commutes bit-exactly). Never VFMADD*: every
// multiply and add rounds separately, exactly like the scalar code.

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

// func FFTColsR4(r0, r1, r2, r3 []complex64, w1, w2, w3 complex64, inverse bool)
// One radix-4 column stage on rows (r, r+h, r+2h, r+3h): per element
// exactly the scalar colsR4Pass sequence — tb = r2*w1, tc = r1*w2,
// td = r3*w3 (each the shared mul idiom, twiddles pre-adjusted for
// direction by the caller), plain-add combine, and the +-i rotation as
// an exact swap + sign-bit xor selected by inverse. All rows share
// r0's length.
// Register map (FFTColsR4):
//   DI=r0  DX=r1  BX=r2  R8=r3  SI=len  CX=len&^3  R10=index  AX=inverse
//   Y0/Y1=w1 re/im  Y2/Y3=w2 re/im  Y4/Y5=w3 re/im  Y15=rotation mask
//   loop: Y6=A  Y7=C->s0  Y8=B->s1  Y9=D->s2  Y10=scratch/swap/s3->rot
//   Y11=tb->out2  Y12=tc->out1  Y13=td->out3
TEXT ·FFTColsR4(SB), NOSPLIT, $0-121
	MOVQ         r0_base+0(FP), DI
	MOVQ         r0_len+8(FP), SI
	MOVQ         r1_base+24(FP), DX
	MOVQ         r2_base+48(FP), BX
	MOVQ         r3_base+72(FP), R8
	VBROADCASTSS w1_real+96(FP), Y0
	VBROADCASTSS w1_imag+100(FP), Y1
	VBROADCASTSS w2_real+104(FP), Y2
	VBROADCASTSS w2_imag+108(FP), Y3
	VBROADCASTSS w3_real+112(FP), Y4
	VBROADCASTSS w3_imag+116(FP), Y5
	MOVBLZX      inverse+120(FP), AX
	// rotation sign mask: forward negates the imag lane (-i*s3),
	// inverse negates the real lane (+i*s3); xor is an exact sign flip
	MOVQ         $0x8000000000000000, R11
	TESTB        AX, AX
	JZ           r4_mask
	MOVQ         $0x0000000080000000, R11

r4_mask:
	MOVQ         R11, X15
	VPBROADCASTQ X15, Y15
	MOVQ         SI, CX
	ANDQ         $-4, CX
	XORQ         R10, R10
	CMPQ         CX, $0
	JEQ          r4_tail

r4_main:
	VMOVUPS   (DI)(R10*8), Y6   // A  = row r
	VMOVUPS   (DX)(R10*8), Y7   // C  = row r+h   (q=2 input)
	VMOVUPS   (BX)(R10*8), Y8   // B  = row r+2h  (q=1 input)
	VMOVUPS   (R8)(R10*8), Y9   // D  = row r+3h  (q=3 input)
	VPERMILPS $0xB1, Y8, Y10
	VMULPS    Y0, Y8, Y11       // t1 = B*w1r
	VMULPS    Y1, Y10, Y10      // t2 = swap(B)*w1i
	VADDSUBPS Y10, Y11, Y11     // tb = B*w1
	VPERMILPS $0xB1, Y7, Y10
	VMULPS    Y2, Y7, Y12
	VMULPS    Y3, Y10, Y10
	VADDSUBPS Y10, Y12, Y12     // tc = C*w2
	VPERMILPS $0xB1, Y9, Y10
	VMULPS    Y4, Y9, Y13
	VMULPS    Y5, Y10, Y10
	VADDSUBPS Y10, Y13, Y13     // td = D*w3
	VADDPS    Y12, Y6, Y7       // s0 = A + tc
	VSUBPS    Y12, Y6, Y8       // s1 = A - tc
	VADDPS    Y13, Y11, Y9      // s2 = tb + td
	VSUBPS    Y13, Y11, Y10     // s3 = tb - td
	VPERMILPS $0xB1, Y10, Y10   // swap(s3)
	VXORPS    Y15, Y10, Y10     // rot = -+i*s3
	VADDPS    Y9, Y7, Y6        // out0 = s0 + s2
	VSUBPS    Y9, Y7, Y11       // out2 = s0 - s2
	VADDPS    Y10, Y8, Y12      // out1 = s1 + rot
	VSUBPS    Y10, Y8, Y13      // out3 = s1 - rot
	VMOVUPS   Y6, (DI)(R10*8)
	VMOVUPS   Y12, (DX)(R10*8)
	VMOVUPS   Y11, (BX)(R10*8)
	VMOVUPS   Y13, (R8)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, CX
	JLT       r4_main

r4_tail:
	CMPQ R10, SI
	JGE  r4_done

r4_tail_loop:
	VMOVSD    (DI)(R10*8), X6
	VMOVSD    (DX)(R10*8), X7
	VMOVSD    (BX)(R10*8), X8
	VMOVSD    (R8)(R10*8), X9
	VPERMILPS $0xB1, X8, X10
	VMULPS    X0, X8, X11
	VMULPS    X1, X10, X10
	VADDSUBPS X10, X11, X11
	VPERMILPS $0xB1, X7, X10
	VMULPS    X2, X7, X12
	VMULPS    X3, X10, X10
	VADDSUBPS X10, X12, X12
	VPERMILPS $0xB1, X9, X10
	VMULPS    X4, X9, X13
	VMULPS    X5, X10, X10
	VADDSUBPS X10, X13, X13
	VADDPS    X12, X6, X7
	VSUBPS    X12, X6, X8
	VADDPS    X13, X11, X9
	VSUBPS    X13, X11, X10
	VPERMILPS $0xB1, X10, X10
	VXORPS    X15, X10, X10
	VADDPS    X9, X7, X6
	VSUBPS    X9, X7, X11
	VADDPS    X10, X8, X12
	VSUBPS    X10, X8, X13
	VMOVSD    X6, (DI)(R10*8)
	VMOVSD    X12, (DX)(R10*8)
	VMOVSD    X11, (BX)(R10*8)
	VMOVSD    X13, (R8)(R10*8)
	INCQ      R10
	CMPQ      R10, SI
	JLT       r4_tail_loop

r4_done:
	VZEROUPPER
	RET

// func FFTColsHead(p, q []complex64)
// The odd-log2 head stage on a row pair: p,q = p+q, p-q. Plain
// single-rounded adds, no twiddles — exactly the scalar colsR4Head.
// Register map (FFTColsHead):
//   DI=p  DX=q  SI=len  CX=len&^3  R10=index
//   loop: Y1=p  Y2=q  Y3=sum  Y4=diff (X forms in the tail)
TEXT ·FFTColsHead(SB), NOSPLIT, $0-48
	MOVQ p_base+0(FP), DI
	MOVQ p_len+8(FP), SI
	MOVQ q_base+24(FP), DX
	MOVQ SI, CX
	ANDQ $-4, CX
	XORQ R10, R10
	CMPQ CX, $0
	JEQ  hd_tail

hd_main:
	VMOVUPS (DI)(R10*8), Y1
	VMOVUPS (DX)(R10*8), Y2
	VADDPS  Y2, Y1, Y3
	VSUBPS  Y2, Y1, Y4
	VMOVUPS Y3, (DI)(R10*8)
	VMOVUPS Y4, (DX)(R10*8)
	ADDQ    $4, R10
	CMPQ    R10, CX
	JLT     hd_main

hd_tail:
	CMPQ R10, SI
	JGE  hd_done

hd_tail_loop:
	VMOVSD (DI)(R10*8), X1
	VMOVSD (DX)(R10*8), X2
	VADDPS X2, X1, X3
	VSUBPS X2, X1, X4
	VMOVSD X3, (DI)(R10*8)
	VMOVSD X4, (DX)(R10*8)
	INCQ   R10
	CMPQ   R10, SI
	JLT    hd_tail_loop

hd_done:
	VZEROUPPER
	RET

// Alternating-qword mask for building the h=1 per-128 rotation mask:
// (0, ~0, 0, ~0) selects the s3 slot of each 128-bit half.
DATA r4alt<>+0(SB)/8, $0x0000000000000000
DATA r4alt<>+8(SB)/8, $0xffffffffffffffff
DATA r4alt<>+16(SB)/8, $0x0000000000000000
DATA r4alt<>+24(SB)/8, $0xffffffffffffffff
GLOBL r4alt<>(SB), RODATA, $32

// func FFTStagesR4(a []complex64, tri []complex64, inverse bool)
// The complete radix-4 cascade over one bit-reversed row (len a power
// of two >= 8): the odd-log2 head stage (pure adds), the h=1 radix-4
// stage on in-register groups (lanes 1..3 go through the w=(1,0)
// multiply idiom, lane 0 stays untouched exactly like the scalar,
// which never multiplies A), the h=2 XMM stage and the generic h>=4
// stage — per element exactly the scalar fftR4 sequence: mulPlain via
// the shared two-products-one-addsub idiom (inverse conjugates the
// twiddles by an exact wi-lane sign flip), plain-add combines, and the
// -+i rotation as swap + sign-bit xor. No FMA anywhere (7.2 contract).
// Register map (FFTStagesR4):
//   DI=a  SI=len  DX=tri cursor  AX=inverse  R8=h  R9=base  R10=j/idx
//   R11=A ptr  R12=C ptr (+h)  R13=B ptr (+2h)  R14=D ptr (+3h)
//   Y15=wi conjugation mask  Y14=full rotation mask  Y13=h1 per-128
//   rotation mask; loop scratch: Y0/Y1=w1 re/im  Y2/Y3=w2 re/im
//   Y4/Y5=w3 re/im  Y6=A  Y7=C->tc/s0  Y8=B->tb/s1  Y9=D->td/s2
//   Y10=swap scratch/s3->rot  Y11/Y12=P,M / out scratch
TEXT ·FFTStagesR4(SB), NOSPLIT, $0-49
	MOVQ    a_base+0(FP), DI
	MOVQ    a_len+8(FP), SI
	MOVQ    tri_base+24(FP), DX
	MOVBLZX inverse+48(FP), AX

	// Y15: wi-lane conjugation mask (0x80000000 everywhere when inverse)
	MOVL AX, R10
	SHLL $31, R10
	MOVD R10, X15
	VPBROADCASTD X15, Y15

	// Y14: rotation mask — forward negates the imag lane of each
	// complex (-i*s3), inverse the real lane (+i*s3)
	MOVQ $0x8000000000000000, R11
	TESTB AX, AX
	JZ   sr4_masks
	MOVQ $0x0000000080000000, R11

sr4_masks:
	MOVQ         R11, X14
	VPBROADCASTQ X14, Y14
	// Y13: h=1 rotation mask — the same qword pattern but only in the
	// s3 slot (qword 1) of each 128-bit half
	VPAND r4alt<>(SB), Y14, Y13

	// ---- head stage (odd log2): adjacent pairs (a,b) -> (a+b, a-b) --
	MOVQ SI, CX          // log2 parity probe: n has a single set bit;
	MOVQ $0xAAAAAAAAAAAAAAAA, R11
	TESTQ R11, CX        // bits 1,3,5,... set => log2(n) odd
	MOVQ $1, R8          // h = 1
	JZ   sr4_h1          // even log2: straight to the h=1 radix-4 stage

	XORQ R10, R10

sr4_head:
	VMOVUPS   (DI)(R10*8), Y6
	VMOVDDUP  Y6, Y7            // (c0, c0, c2, c2)
	VPERMILPD $0x0F, Y6, Y8     // (c1, c1, c3, c3)
	VADDPS    Y8, Y7, Y9        // sums
	VSUBPS    Y8, Y7, Y10       // diffs
	VBLENDPS  $0xCC, Y10, Y9, Y6
	VMOVUPS   Y6, (DI)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, SI
	JLT       sr4_head

	MOVQ $2, R8                 // radix-4 stages start at h = 2
	JMP  sr4_stage

	// ---- h=1 radix-4 stage: one YMM per 4-complex group ------------
	// Lanes hold (A, Cin, Bin, Din); twiddles are all (1,0). Lanes
	// 1..3 run the multiply idiom (wr=1 lanes, wi=0^conj lanes) and
	// lane 0 is blended back unmultiplied, matching the scalar exactly
	// (mulPlain(v, 1) is not a bitwise no-op on -0 inputs).
sr4_h1:
	ADDQ         $24, DX        // skip the h=1 tri block (3 entries) —
	                            // this stage's twiddles are hardcoded 1
	VXORPS       Y1, Y1, Y1     // wi = 0
	VXORPS       Y15, Y1, Y1    // conj sign on the zero lanes (inverse: -0)
	XORQ         R10, R10
	MOVL         $0x3F800000, R11
	MOVD         R11, X0
	VPBROADCASTD X0, Y0         // wr = 1 lanes

sr4_h1_loop:
	VMOVUPS   (DI)(R10*8), Y6
	VPERMILPS $0xB1, Y6, Y10
	VMULPS    Y0, Y6, Y11       // t1 = v*1
	VMULPS    Y1, Y10, Y10      // t2 = swap(v)*(+-0)
	VADDSUBPS Y10, Y11, Y11     // mulPlain(v, (1, +-0)) all lanes
	VBLENDPS  $0x03, Y6, Y11, Y6 // lane 0 back to raw A
	// butterfly across lanes: V = (A, tc, tb, td)
	VMOVDDUP  Y6, Y7            // (A, A, tb, tb)
	VPERMILPD $0x0F, Y6, Y8     // (tc, tc, td, td)
	VADDPS    Y8, Y7, Y11       // (s0, s0, s2, s2)
	VSUBPS    Y8, Y7, Y12       // (s1, s1, s3, s3)
	VBLENDPS  $0xCC, Y12, Y11, Y6 // W = (s0, s1, s2, s3)
	VPERM2F128 $0x00, Y6, Y6, Y7 // U = (s0, s1, s0, s1)
	VPERM2F128 $0x11, Y6, Y6, Y8 // T = (s2, s3, s2, s3)
	VPERMILPS $0xB4, Y8, Y8     // s3 slots pair-swapped
	VXORPS    Y13, Y8, Y8       // T' = (s2, rot, s2, rot)
	VADDPS    Y8, Y7, Y11       // (s0+s2, s1+rot, ..)
	VSUBPS    Y8, Y7, Y12       // (.., .., s0-s2, s1-rot)
	VBLENDPS  $0xF0, Y12, Y11, Y6
	VMOVUPS   Y6, (DI)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, SI
	JLT       sr4_h1_loop

	MOVQ $4, R8                 // next radix-4 stage: h = 4

	// ---- radix-4 stage loop (h = 2 via XMM, h >= 4 via YMM) ---------
sr4_stage:
	MOVQ R8, CX
	SHLQ $2, CX
	CMPQ CX, SI                 // 4h <= n ?
	JGT  sr4_done

	XORQ R9, R9                 // base = 0
	CMPQ R8, $2
	JEQ  sr4_h2_groups

sr4_groups:
	LEAQ (DI)(R9*8), R11        // A = a + base
	LEAQ (R11)(R8*8), R12       // C = A + h
	LEAQ (R12)(R8*8), R13       // B = A + 2h
	LEAQ (R13)(R8*8), R14       // D = A + 3h
	XORQ R10, R10               // j = 0

sr4_j:
	// twiddle chunks: w1 at tri[j], w2 at tri[h+j], w3 at tri[2h+j]
	VMOVUPS   (DX)(R10*8), Y0
	VMOVSHDUP Y0, Y1
	VMOVSLDUP Y0, Y0
	VXORPS    Y15, Y1, Y1
	LEAQ      (DX)(R8*8), CX
	VMOVUPS   (CX)(R10*8), Y2
	VMOVSHDUP Y2, Y3
	VMOVSLDUP Y2, Y2
	VXORPS    Y15, Y3, Y3
	LEAQ      (CX)(R8*8), CX
	VMOVUPS   (CX)(R10*8), Y4
	VMOVSHDUP Y4, Y5
	VMOVSLDUP Y4, Y4
	VXORPS    Y15, Y5, Y5

	VMOVUPS (R11)(R10*8), Y6    // A
	VMOVUPS (R12)(R10*8), Y7    // C-input
	VMOVUPS (R13)(R10*8), Y8    // B-input
	VMOVUPS (R14)(R10*8), Y9    // D-input
	VPERMILPS $0xB1, Y8, Y10
	VMULPS    Y0, Y8, Y11
	VMULPS    Y1, Y10, Y10
	VADDSUBPS Y10, Y11, Y8      // tb = B*w1
	VPERMILPS $0xB1, Y7, Y10
	VMULPS    Y2, Y7, Y11
	VMULPS    Y3, Y10, Y10
	VADDSUBPS Y10, Y11, Y7      // tc = C*w2
	VPERMILPS $0xB1, Y9, Y10
	VMULPS    Y4, Y9, Y11
	VMULPS    Y5, Y10, Y10
	VADDSUBPS Y10, Y11, Y9      // td = D*w3
	VADDPS    Y7, Y6, Y11       // s0
	VSUBPS    Y7, Y6, Y12       // s1
	VADDPS    Y9, Y8, Y7        // s2
	VSUBPS    Y9, Y8, Y10       // s3
	VPERMILPS $0xB1, Y10, Y10
	VXORPS    Y14, Y10, Y10     // rot
	VADDPS    Y7, Y11, Y6       // out0 = s0+s2
	VSUBPS    Y7, Y11, Y8       // out2 = s0-s2
	VADDPS    Y10, Y12, Y7      // out1 = s1+rot
	VSUBPS    Y10, Y12, Y9      // out3 = s1-rot
	VMOVUPS   Y6, (R11)(R10*8)
	VMOVUPS   Y7, (R12)(R10*8)
	VMOVUPS   Y8, (R13)(R10*8)
	VMOVUPS   Y9, (R14)(R10*8)
	ADDQ      $4, R10
	CMPQ      R10, R8
	JLT       sr4_j

	LEAQ (R9)(R8*4), R9         // base += 4h
	CMPQ R9, SI
	JLT  sr4_groups
	JMP  sr4_next

	// h=2: quarters are 2 complexes apart — one XMM iteration per group
sr4_h2_groups:
	VMOVUPS   (DX), X0          // w1[0..1]
	VMOVSHDUP X0, X1
	VMOVSLDUP X0, X0
	VXORPS    X15, X1, X1
	VMOVUPS   16(DX), X2        // w2[0..1]
	VMOVSHDUP X2, X3
	VMOVSLDUP X2, X2
	VXORPS    X15, X3, X3
	VMOVUPS   32(DX), X4        // w3[0..1]
	VMOVSHDUP X4, X5
	VMOVSLDUP X4, X4
	VXORPS    X15, X5, X5

sr4_h2_loop:
	LEAQ      (DI)(R9*8), R11
	VMOVUPS   (R11), X6         // A pair
	VMOVUPS   16(R11), X7       // C pair
	VMOVUPS   32(R11), X8       // B pair
	VMOVUPS   48(R11), X9       // D pair
	VPERMILPS $0xB1, X8, X10
	VMULPS    X0, X8, X11
	VMULPS    X1, X10, X10
	VADDSUBPS X10, X11, X8
	VPERMILPS $0xB1, X7, X10
	VMULPS    X2, X7, X11
	VMULPS    X3, X10, X10
	VADDSUBPS X10, X11, X7
	VPERMILPS $0xB1, X9, X10
	VMULPS    X4, X9, X11
	VMULPS    X5, X10, X10
	VADDSUBPS X10, X11, X9
	VADDPS    X7, X6, X11
	VSUBPS    X7, X6, X12
	VADDPS    X9, X8, X7
	VSUBPS    X9, X8, X10
	VPERMILPS $0xB1, X10, X10
	VXORPS    X14, X10, X10
	VADDPS    X7, X11, X6
	VSUBPS    X7, X11, X8
	VADDPS    X10, X12, X7
	VSUBPS    X10, X12, X9
	VMOVUPS   X6, (R11)
	VMOVUPS   X7, 16(R11)
	VMOVUPS   X8, 32(R11)
	VMOVUPS   X9, 48(R11)
	ADDQ      $8, R9            // next group (8 complexes)
	CMPQ      R9, SI
	JLT       sr4_h2_loop

sr4_next:
	LEAQ (DX)(R8*8), DX         // tri += 3h entries
	LEAQ (DX)(R8*8), DX
	LEAQ (DX)(R8*8), DX
	SHLQ $2, R8                 // h *= 4
	JMP  sr4_stage

sr4_done:
	VZEROUPPER
	RET
