//go:build gc && !purego

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

// func NormRow(rrow []float32, crow []float32, wt *float32, stride, n int,
//              numScale, varScale, eps, templNorm float32)
// One float32 kernel for every channel count: the spill already folded
// the channels into the cross/idiff/s2 lanes (exact integers, converted
// once), so the tail runs 8 lanes per iteration with no cn dispatch and
// no f64 anywhere. Sequence per lane = the scalar normOne exactly:
// mul/sub/sqrt/div correctly rounded, predicates and blends exact — and
// deliberately NO FMA: fusing cross*invArea into the subtract would
// break asm<->purego bit-identity (Phase 7 ledger, fma-everywhere).
// Register map (NormRow):
//   DI=rrow  SI=crow  R8=wt(cross)  R9=stride(bytes)  R10=idiff lane
//   R11=s2 lane  DX=n  AX=index
//   Y9=0.5  Y10=1.0  Y11=1.125  Y12=numScale  Y8=varScale  Y13=eps
//   Y14=templNorm
//   Y15=abs mask; loop scratch: Y0=num  Y1=diff2/numer/dv  Y2=sq/scratch
//   Y3=den  Y4=s2/m1  Y5=lim/an/m2
TEXT ·NormRow(SB), NOSPLIT, $0-88
	MOVQ rrow_base+0(FP), DI
	MOVQ crow_base+24(FP), SI
	MOVQ wt+48(FP), R8          // cross lane
	MOVQ stride+56(FP), R9
	SHLQ $2, R9                 // stride in bytes
	MOVQ n+64(FP), DX

	VBROADCASTSS normconst<>+0(SB), Y9   // 0.5
	VBROADCASTSS normconst<>+4(SB), Y10  // 1.0
	VBROADCASTSS normconst<>+8(SB), Y11  // 1.125
	VBROADCASTSS numScale+72(FP), Y12
	VBROADCASTSS varScale+76(FP), Y8
	VBROADCASTSS eps+80(FP), Y13
	VBROADCASTSS templNorm+84(FP), Y14
	VMOVUPS      absmask<>(SB), Y15

	LEAQ (R8)(R9*1), R10        // idiff lane
	LEAQ (R10)(R9*1), R11       // s2 lane
	XORQ AX, AX                 // element index

norm_loop:
	CMPQ AX, DX
	JGE  norm_done
	VMOVUPS   (R8)(AX*4), Y1    // lane0
	VMULPS    Y12, Y1, Y1
	VMOVUPS   (SI)(AX*4), Y0
	VSUBPS    Y1, Y0, Y0        // num = corr - lane0*numScale
	VMOVUPS   (R10)(AX*4), Y1
	VMULPS    Y8, Y1, Y1        // diff2 = idiff*varScale (>= +0: both nonneg)
	VMOVUPS   (R11)(AX*4), Y4   // s2
	VMULPS    Y13, Y4, Y5       // e = eps*s2
	VMINPS    Y5, Y9, Y5        // lim = 0.5 < e ? 0.5 : e
	VSQRTPS   Y1, Y2
	VMULPS    Y14, Y2, Y2       // sq = sqrt(diff2)*templNorm
	VCMPPS    $0x12, Y5, Y1, Y3 // diff2 <= lim (LE_OQ)
	VANDNPS   Y2, Y3, Y3        // den = mask ? 0 : sq
	VANDPS    Y15, Y0, Y5       // an = |num|
	VCMPPS    $0x11, Y3, Y5, Y4 // m1 = an < den (LT_OQ)
	VANDPS    Y4, Y0, Y1        // numer = m1 ? num : 0
	VBLENDVPS Y4, Y3, Y10, Y2   // divisor = m1 ? den : 1.0
	VDIVPS    Y2, Y1, Y1        // dv
	VMULPS    Y11, Y3, Y2       // den*1.125
	VCMPPS    $0x11, Y2, Y5, Y5 // m2 = an < den*1.125
	VANDPS    signmask<>(SB), Y0, Y2
	VORPS     Y10, Y2, Y2       // sat = copysign(1, num); only selected
	                            // when num is nonzero, where it equals
	                            // the scalar num>0 ? 1 : -1
	VANDPS    Y5, Y2, Y2        // inner = m2 ? sat : 0
	VBLENDVPS Y4, Y1, Y2, Y2    // v = m1 ? dv : inner
	VMOVUPS   Y2, (DI)(AX*4)
	ADDQ      $8, AX
	JMP       norm_loop

norm_done:
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

DATA absmask<>+0(SB)/8, $0x7fffffff7fffffff
DATA absmask<>+8(SB)/8, $0x7fffffff7fffffff
DATA absmask<>+16(SB)/8, $0x7fffffff7fffffff
DATA absmask<>+24(SB)/8, $0x7fffffff7fffffff
GLOBL absmask<>(SB), RODATA|NOPTR, $32

DATA signmask<>+0(SB)/8, $0x8000000080000000
DATA signmask<>+8(SB)/8, $0x8000000080000000
DATA signmask<>+16(SB)/8, $0x8000000080000000
DATA signmask<>+24(SB)/8, $0x8000000080000000
GLOBL signmask<>(SB), RODATA|NOPTR, $32

DATA normconst<>+0(SB)/4, $0x3f000000 // 0.5f
DATA normconst<>+4(SB)/4, $0x3f800000 // 1.0f
DATA normconst<>+8(SB)/4, $0x3f900000 // 1.125f
GLOBL normconst<>(SB), RODATA|NOPTR, $12

// func SpillStats1(wt []float32, stride int, lo, hi []int32, lo2, hi2 []int64,
//	s0, s2, area int64) (ns0, ns2 int64)
// The cn=1 normalize spill, 4 elements per pass. Everything integer is
// exact: window sums stay below 2^31 (caller-gated area bound), so every wide
// product decomposes into one signed 32x32->64 VPMULDQ, the running
// idiff advances by Didiff = area*d2 - 2*d*s0 - d*d (algebra of
// (s+d)^2), and int64 prefix-sum regrouping is free. Each emitted lane
// is float32(float64(v)): float64 built exactly as hi*2^32 + lo (both
// addends exact, one rounding) or a single 2^52 bithack when v < 2^52,
// then one VCVTPD2PS — the same two roundings as the scalar spill.
// Register map (SpillStats1):
//   DI=wt(s0 lane) R10=idiff lane R11=s2 lane R9=stride(bytes) DX=n SI=i
//   R8=lo BX=hi R12=lo2 R13=hi2 AX=s0 CX=s2 R14=idiff0 R15=scratch
//   Y15=2^52 magic Y14=2^32(f64) Y13=low-32 mask Y11=area
//   loop: Y0=d Y1/Y3=slide scratch Y2=inclusive prefix Y4=s0v
//   Y6=Didiff/idiff64 Y7=d2 Y8=t2v/products Y9/Y10=converts
TEXT ·SpillStats1(SB), NOSPLIT, $0-168
	MOVQ wt_base+0(FP), DI
	MOVQ wt_len+8(FP), DX
	MOVQ stride+24(FP), R9
	SHLQ $2, R9                 // stride in bytes
	MOVQ lo_base+32(FP), R8
	MOVQ hi_base+56(FP), BX
	MOVQ lo2_base+80(FP), R12
	MOVQ hi2_base+104(FP), R13
	MOVQ s0+128(FP), AX
	MOVQ s2+136(FP), CX
	VPBROADCASTQ magic52<>(SB), Y15
	VBROADCASTSD pow32<>(SB), Y14
	VPBROADCASTQ mask32lo<>(SB), Y13
	VPBROADCASTQ area+144(FP), Y11
	LEAQ (DI)(R9*1), R10        // idiff lane
	LEAQ (R10)(R9*1), R11       // s2 lane
	XORQ SI, SI

sp1_loop:
	CMPQ SI, DX
	JGE  sp1_done
	// idiff0 = area*s2 - s0*s0 at quad entry (scalar, exact)
	MOVQ  CX, R14
	IMULQ area+144(FP), R14
	MOVQ  AX, R15
	IMULQ AX, R15
	SUBQ  R15, R14
	// d = hi-lo as int64 lanes; exclusive prefix -> per-element s0v
	VPMOVSXDQ (BX)(SI*4), Y0
	VPMOVSXDQ (R8)(SI*4), Y1
	VPSUBQ    Y1, Y0, Y0        // d
	VPERMQ    $0x93, Y0, Y1
	VPAND     slidem1<>(SB), Y1, Y1
	VPADDQ    Y1, Y0, Y2
	VPERMQ    $0x4E, Y2, Y3
	VPAND     slidem2<>(SB), Y3, Y3
	VPADDQ    Y3, Y2, Y2        // inclusive prefix of d
	VPERMQ    $0x93, Y2, Y3
	VPAND     slidem1<>(SB), Y3, Y3
	VMOVQ     AX, X4
	VPBROADCASTQ X4, Y4
	VPADDQ    Y3, Y4, Y4        // s0v (pre-slide, < 2^31)
	VEXTRACTI128 $1, Y2, X3
	VPEXTRQ   $1, X3, R15
	ADDQ      R15, AX           // s0 advanced past the quad
	// s0 lane: s0v < 2^31 < 2^52 -> one bithack is exactly float64(s0v)
	VPOR      Y15, Y4, Y9
	VSUBPD    Y15, Y9, Y9
	VCVTPD2PSY Y9, X9
	VMOVUPS   X9, (DI)(SI*4)
	// Didiff = area*d2 - 2*d*s0v - d*d
	VMOVDQU   (R13)(SI*8), Y7
	VMOVDQU   (R12)(SI*8), Y8
	VPSUBQ    Y8, Y7, Y7        // d2 (|d2| < 2^31: th gate)
	VPMULDQ   Y11, Y7, Y6       // area*d2
	VPMULDQ   Y0, Y4, Y8        // d*s0v
	VPSLLQ    $1, Y8, Y8
	VPSUBQ    Y8, Y6, Y6
	VPMULDQ   Y0, Y0, Y8        // d*d
	VPSUBQ    Y8, Y6, Y6        // Didiff
	// t2v = s2 + exclusive prefix of d2; advance scalar s2
	VPERMQ    $0x93, Y7, Y1
	VPAND     slidem1<>(SB), Y1, Y1
	VPADDQ    Y1, Y7, Y2
	VPERMQ    $0x4E, Y2, Y3
	VPAND     slidem2<>(SB), Y3, Y3
	VPADDQ    Y3, Y2, Y2
	VPERMQ    $0x93, Y2, Y3
	VPAND     slidem1<>(SB), Y3, Y3
	VMOVQ     CX, X8
	VPBROADCASTQ X8, Y8
	VPADDQ    Y3, Y8, Y8        // t2v
	VEXTRACTI128 $1, Y2, X3
	VPEXTRQ   $1, X3, R15
	ADDQ      R15, CX
	// idiff64 = idiff0 + exclusive prefix of Didiff
	VPERMQ    $0x93, Y6, Y1
	VPAND     slidem1<>(SB), Y1, Y1
	VPADDQ    Y1, Y6, Y2
	VPERMQ    $0x4E, Y2, Y3
	VPAND     slidem2<>(SB), Y3, Y3
	VPADDQ    Y3, Y2, Y2
	VPERMQ    $0x93, Y2, Y3
	VPAND     slidem1<>(SB), Y3, Y3
	VMOVQ     R14, X6
	VPBROADCASTQ X6, Y6
	VPADDQ    Y3, Y6, Y6        // idiff64 (>= 0, < 2^62)
	// s2 lane: t2v < 2^52 -> one bithack is exactly float64(t2v)
	VPOR      Y15, Y8, Y9
	VSUBPD    Y15, Y9, Y9
	VCVTPD2PSY Y9, X9
	VMOVUPS   X9, (R11)(SI*4)
	// idiff lane: hi*2^32 + lo (exact addends, one rounding) -> f32
	VPSRLQ    $32, Y6, Y9
	VPAND     Y13, Y6, Y10
	VPOR      Y15, Y9, Y9
	VSUBPD    Y15, Y9, Y9
	VPOR      Y15, Y10, Y10
	VSUBPD    Y15, Y10, Y10
	VMULPD    Y14, Y9, Y9
	VADDPD    Y10, Y9, Y9
	VCVTPD2PSY Y9, X9
	VMOVUPS   X9, (R10)(SI*4)
	ADDQ      $4, SI
	JMP       sp1_loop

sp1_done:
	MOVQ AX, ns0+152(FP)
	MOVQ CX, ns2+160(FP)
	VZEROUPPER
	RET

DATA magic52<>+0(SB)/8, $0x4330000000000000
GLOBL magic52<>(SB), RODATA|NOPTR, $8

DATA pow32<>+0(SB)/8, $0x41F0000000000000 // 2^32 as float64
GLOBL pow32<>(SB), RODATA|NOPTR, $8

DATA mask32lo<>+0(SB)/8, $0x00000000FFFFFFFF
GLOBL mask32lo<>(SB), RODATA|NOPTR, $8

DATA slidem1<>+0(SB)/8, $0
DATA slidem1<>+8(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA slidem1<>+16(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA slidem1<>+24(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL slidem1<>(SB), RODATA|NOPTR, $32

DATA slidem2<>+0(SB)/8, $0
DATA slidem2<>+8(SB)/8, $0
DATA slidem2<>+16(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA slidem2<>+24(SB)/8, $0xFFFFFFFFFFFFFFFF
GLOBL slidem2<>(SB), RODATA|NOPTR, $32

// func SpillStats4(wt []float32, stride int, lo, hi []int32, lo2, hi2 []int64,
//                  s *[4]int64, tsum *[4]int64, s2, area int64, four bool) int64
// The cn=3/4 normalize spill over len(wt) elements (a multiple of 4 by
// the caller's vns mask, though any length works), RGBA column-sum
// layout (4 int32 lanes per pixel; the caller gates cs == 4 and
// cn·255²·th < 2^31 so |Δ colSum2| is a 32-bit value; the per-cn stats
// caps bound every window sum below 2^31 and every cross/idiff below
// 2^63). Pixel-major: the four channel sums live in one int64x4
// register that slides by one VPMOVSXDQ+VPADDQ per element; cross and
// Σ sk² come from exact VPMULDQ products and horizontal adds; the
// float32 lanes are CVTSI2SDQ then CVTSD2SS — precisely the scalar
// float32(float64(v)) two-rounding sequence, so output is
// bit-identical to the cn=3/4 Go loops on every gated input. cn=3
// zeroes the alpha lanes of both vectors so s3 never moves and
// contributes nothing. s[0..3] advance in place; advanced s2 returns.
// Register map (SpillStats4):
//   DI=wt(cross lane) R10=idiff lane R11=s2 lane R9=stride(bytes)
//   DX=n SI=i R8=lo BX=hi R12=lo2 R13=hi2 CX=s2 AX=s ptr
//   R14=cross/idiff scratch R15=d2/scratch
//   Y4=s_vec Y5=t_vec Y3=alpha mask; loop: Y0=d Y1=products X6/X7=hsum
TEXT ·SpillStats4(SB), NOSPLIT, $0-176
	MOVQ wt_base+0(FP), DI
	MOVQ wt_len+8(FP), DX
	MOVQ stride+24(FP), R9
	SHLQ $2, R9
	MOVQ lo_base+32(FP), R8
	MOVQ hi_base+56(FP), BX
	MOVQ lo2_base+80(FP), R12
	MOVQ hi2_base+104(FP), R13
	MOVQ s2+144(FP), CX
	MOVQ s+128(FP), AX
	VMOVDQU (AX), Y4            // s_vec = (s0, s1, s2c, s3)
	MOVQ tsum+136(FP), R14
	VMOVDQU (R14), Y5           // t_vec
	// cn=3: force the alpha lanes of both vectors to zero — s3 then
	// never moves and neither dot product sees it
	VPCMPEQQ Y3, Y3, Y3
	MOVBLZX  four+160(FP), R15
	TESTB    R15, R15
	JNZ      sp4_masked
	VMOVDQU  sp4rgb<>(SB), Y3   // (~0, ~0, ~0, 0)
sp4_masked:
	VPAND Y3, Y4, Y4
	VPAND Y3, Y5, Y5
	LEAQ (DI)(R9*1), R10
	LEAQ (R10)(R9*1), R11
	XORQ SI, SI

sp4_loop:
	CMPQ SI, DX
	JGE  sp4_done
	// cross = Σ sk*tk (exact: sums and tsums < 2^31 by the caps)
	VPMULDQ Y5, Y4, Y1
	VEXTRACTI128 $1, Y1, X6
	VPADDQ  X6, X1, X6
	VPSHUFD $0x4E, X6, X7
	VPADDQ  X7, X6, X6
	VMOVQ   X6, R14
	VCVTSI2SDQ R14, X6, X6
	VCVTSD2SS X6, X6, X6
	VMOVSS  X6, (DI)(SI*4)
	// idiff = area*s2 - Σ sk*sk
	VPMULDQ Y4, Y4, Y1
	VEXTRACTI128 $1, Y1, X6
	VPADDQ  X6, X1, X6
	VPSHUFD $0x4E, X6, X7
	VPADDQ  X7, X6, X6
	VMOVQ   X6, R15
	MOVQ    CX, R14
	IMULQ   area+152(FP), R14
	SUBQ    R15, R14
	VCVTSI2SDQ R14, X6, X6
	VCVTSD2SS X6, X6, X6
	VMOVSS  X6, (R10)(SI*4)
	// s2 lane, then slide: s2 += hi2-lo2, s_vec += (hi-lo) per channel
	VCVTSI2SDQ CX, X6, X6
	VCVTSD2SS X6, X6, X6
	VMOVSS  X6, (R11)(SI*4)
	MOVQ    (R13)(SI*8), R15
	SUBQ    (R12)(SI*8), R15
	ADDQ    R15, CX
	LEAQ    (SI)(SI*1), R15
	VPMOVSXDQ (BX)(R15*8), Y0
	VPMOVSXDQ (R8)(R15*8), Y1
	VPSUBQ  Y1, Y0, Y0
	VPAND   Y3, Y0, Y0          // cn=3: alpha delta stays zero
	VPADDQ  Y0, Y4, Y4
	INCQ    SI
	JMP     sp4_loop

sp4_done:
	MOVQ AX, R14                // s ptr still in AX
	VMOVDQU Y4, (R14)
	MOVQ CX, ret+168(FP)
	VZEROUPPER
	RET

DATA sp4rgb<>+0(SB)/8, $0xffffffffffffffff
DATA sp4rgb<>+8(SB)/8, $0xffffffffffffffff
DATA sp4rgb<>+16(SB)/8, $0xffffffffffffffff
DATA sp4rgb<>+24(SB)/8, $0x0000000000000000
GLOBL sp4rgb<>(SB), RODATA|NOPTR, $32
