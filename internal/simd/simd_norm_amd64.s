//go:build gc && cvmatch_asm

#include "textflag.h"
// Normalization and scan kernels: sliding integer column sums, the
// window-sum spill, the branchless normalize tail (sqrt/div/guards), the
// first-occurrence min/max scan, and RGBA->gray.

// Part of the AVX2 kernel set — bit-identical to the generic Go loops.
// Shared conventions (see simd_amd64.s for the CPU-detection entry):
// complex64 values are interleaved (re, im) float32 pairs; the butterfly
// evaluates t1 = q*wr, t2 = swap(q)*wi and addsub(t1, t2) =
// (t1e-t2e, t1o+t2o), which reproduces the scalar complex multiply's two
// products and single rounded add/sub per component (the imaginary sum
// is commuted; IEEE addition commutes bit-exactly). Never VFMADD*: every
// multiply and add rounds separately, exactly like the scalar code.

// func NormRow(rrow []float32, crow []float32, wt *float64, stride, n, cn int,
//              mean *[4]float64, invArea, eps, templNorm float64)
// Register map: constants Y9..Y15 = 0.5, 1.0, 1.125, invArea, eps,
// templNorm, m0; m1..m3 live in Y6..Y8 when cn needs them.
// Register map (NormRow):
//   DI=rrow  SI=crow  R8=wt(t0)  R9=stride(bytes)  R10=t1|q2(cn1)
//   R11=t2  R12=t3|q2(cn3)  R13=q2(cn4)  DX=n  CX=cn  BX=mean  AX=index
//   Y9=0.5  Y10=1.0  Y11=1.125  Y12=invArea  Y13=eps  Y14=templNorm
//   Y15=mean[0]  Y6/Y7/Y8=mean[1..3] (cn>=3)
//   loop scratch: Y0=num  Y1=t_k lanes  Y2=mul scratch  Y3=wndMean2
//   Y4/Y5=den/guard lanes (see norm_tail4)
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
