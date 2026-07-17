// NEON kernel bodies for internal/simd on arm64. Assembled by gen.py with
// clang and spliced into simd_arm64.s as WORD streams (Go's assembler has
// no un-fused vector FP arithmetic).
//
// Contracts mirror the AVX2 kernels instruction-for-instruction in effect:
// every float op is a distinct IEEE single-rounded instruction (fmul/fadd/
// fsub/fdiv/fsqrt/fcvt — never fmla/fmls), negation-by-xor and x+(-y)
// stand in for x86's addsub exactly as on amd64, and integer arithmetic
// may reassociate freely (exact). AArch64 keeps denormals and IEEE
// rounding by default (FPCR.FZ=0), so semantics match the scalar Go loops
// bit for bit.
//
// Conventions (enforced by gen.py):
//   - arguments arrive in x0..x7 / v0..v5 exactly as documented per kernel
//     (the Go asm template loads them from the frame)
//   - only x0-x15 and v0-v31 are touched; no stack, no x18/x26-x30
//   - constants are register-built (mov/movk/movi/fmov) — literal pools
//     would emit relocations, which the generator rejects
//   - each body ends with a single ret (dropped at splice time)
//
// Shared complex-multiply idiom, exactly the scalar bfly/mulConj ops:
//   wr = twiddle real in every lane, wi = twiddle imag in every lane
//   t1 = v * wr                        (per-lane fmul)
//   t2 = rev64(v) * wi                 (re/im swapped, per-lane fmul)
//   addsub(t1, t2) = t1 + (t2 ^ evensign)   -- (t1.e-t2.e, t1.o+t2.o)
// evensign = 0x80000000 in even 32-bit lanes; the identity x-y == x+(-y)
// is IEEE-exact, as already relied on by the amd64 MulConj/Untangle.

// Normalization and scan kernels: sliding integer column sums, the
// window-sum spill, the branchless normalize tail, the first-occurrence
// min/max scan, and RGBA->gray.

.text

// ---------------------------------------------------------------------------
// NormRow: x0=rrow, x1=crow, x2=wt (lane0), x3=stride, x4=n (multiple
// of 8, > 0); v0.s[0]=numScale, v1.s[0]=varScale, v2.s[0]=eps,
// v3.s[0]=templNorm.
// One float32 kernel for every channel count — the spill folded the
// channels into the cross/idiff/s2 lanes (exact integers, converted
// once). Two f32x4 halves (A/B) per pass, mirroring the amd64 kernel op
// for op: mul/sub/fsqrt/fdiv correctly rounded, fcmgt/and/bsl exact.
// Deliberately no fmla: fusing cross*invArea into the subtract would
// break asm<->purego bit-identity (Phase 7 ledger, fma-everywhere).
// Register map: v28=invArea v29=eps v30=templNorm v27=0.5 v25=1.0
// v24=1.125 v23=sign mask; x7=stride bytes, x9/x10=idiff/s2 lane bases,
// x8=i, x11=lane addressing scratch. Per half (A/B): v3/v13=num,
// v5/v15=diff2->numer->dv, v6/v16=lim->m2, v7/v17=den, v8/v18=sat/inner,
// v9/v19=guard mask->result, v10/v20=|num|, v11/v21=m1, v12=divisor.
.global NormRow
NormRow:
	dup	v28.4s, v0.s[0]         // numScale
	dup	v26.4s, v1.s[0]         // varScale
	dup	v29.4s, v2.s[0]         // eps
	dup	v30.4s, v3.s[0]         // templNorm
	fmov	v27.4s, #0.5
	fmov	v25.4s, #1.0
	fmov	v24.4s, #1.125
	mov	w9, #0x80000000
	dup	v23.4s, w9              // f32 sign mask
	lsl	x7, x3, #2              // stride in bytes
	add	x9, x2, x7              // idiff lane base
	add	x10, x9, x7             // s2 lane base
	mov	x8, #0                  // i
.Lnr_loop:
	add	x11, x2, x8, lsl #2
	ldp	q5, q15, [x11]          // lane0 A, B
	fmul	v5.4s, v5.4s, v28.4s    // lane0*numScale
	fmul	v15.4s, v15.4s, v28.4s
	ldp	q3, q13, [x1], #32      // corr A, B
	fsub	v3.4s, v3.4s, v5.4s     // num = corr - lane0*numScale
	fsub	v13.4s, v13.4s, v15.4s
	add	x11, x9, x8, lsl #2
	ldp	q5, q15, [x11]          // idiff A, B
	fmul	v5.4s, v5.4s, v26.4s    // diff2 = idiff*varScale (>= +0)
	fmul	v15.4s, v15.4s, v26.4s
	add	x11, x10, x8, lsl #2
	ldp	q8, q18, [x11]          // s2 A, B
	fmul	v6.4s, v8.4s, v29.4s    // e = eps*s2
	fmul	v16.4s, v18.4s, v29.4s
	fmin	v6.4s, v6.4s, v27.4s    // lim = min(e, 0.5)
	fmin	v16.4s, v16.4s, v27.4s
	fsqrt	v7.4s, v5.4s
	fsqrt	v17.4s, v15.4s
	fmul	v7.4s, v7.4s, v30.4s    // sq = sqrt(diff2)*templNorm
	fmul	v17.4s, v17.4s, v30.4s
	fcmgt	v9.4s, v5.4s, v6.4s     // diff2 > lim
	fcmgt	v19.4s, v15.4s, v16.4s
	and	v7.16b, v7.16b, v9.16b  // den, or 0 below the guard
	and	v17.16b, v17.16b, v19.16b
	fabs	v10.4s, v3.4s           // an = |num| (bit op, exact)
	fabs	v20.4s, v13.4s
	fcmgt	v11.4s, v7.4s, v10.4s   // m1 = an < den
	fcmgt	v21.4s, v17.4s, v20.4s
	mov	v12.16b, v11.16b
	bsl	v12.16b, v7.16b, v25.16b // divisor = m1 ? den : 1.0
	and	v5.16b, v3.16b, v11.16b // m1 ? num : 0
	fdiv	v5.4s, v5.4s, v12.4s    // dv A
	mov	v12.16b, v21.16b
	bsl	v12.16b, v17.16b, v25.16b
	and	v15.16b, v13.16b, v21.16b
	fdiv	v15.4s, v15.4s, v12.4s  // dv B
	fmul	v6.4s, v7.4s, v24.4s    // den*1.125
	fcmgt	v6.4s, v6.4s, v10.4s    // m2 = an < den*1.125
	fmul	v16.4s, v17.4s, v24.4s
	fcmgt	v16.4s, v16.4s, v20.4s
	and	v8.16b, v3.16b, v23.16b
	orr	v8.16b, v8.16b, v25.16b // sat = copysign(1, num)
	and	v8.16b, v8.16b, v6.16b  // inner = m2 ? sat : 0
	and	v18.16b, v13.16b, v23.16b
	orr	v18.16b, v18.16b, v25.16b
	and	v18.16b, v18.16b, v16.16b
	mov	v9.16b, v11.16b
	bsl	v9.16b, v5.16b, v8.16b  // m1 ? dv : inner
	mov	v19.16b, v21.16b
	bsl	v19.16b, v15.16b, v18.16b
	stp	q9, q19, [x0], #32
	add	x8, x8, #8
	cmp	x8, x4
	b.lt	.Lnr_loop
	ret

// ---------------------------------------------------------------------------
// MinMaxRow: x0=row_base, x1=len (nonzero multiple of 8, < 2^31).
// Returns first-occurrence extrema: minV in v0.s[0], maxV in v1.s[0],
// minIdx in v2.s[0], maxIdx in v3.s[0] (the template stores them).
// Per-lane strict ordered compares keep the first occurrence within a
// lane; the reduction breaks value ties by smaller index, so the result
// is the lexicographic (value, index) extremum — the scalar scan's answer.
.global MinMaxRow
MinMaxRow:
	ldp	q0, q16, [x0]           // tracker A: row[0..3], B: row[4..7]
	mov	v1.16b, v0.16b          // maxV A
	mov	v17.16b, v16.16b        // maxV B
	mov	x9, #0x0000000100000000 // idx lanes (0, 1)
	mov	x10, #0x00000002
	movk	x10, #0x3, lsl #32      // idx lanes (2, 3)
	mov	v2.d[0], x9             // minIdx A = (0,1,2,3)
	mov	v2.d[1], x10
	movi	v5.4s, #4
	add	v18.4s, v2.4s, v5.4s    // minIdx B = (4,5,6,7)
	mov	v3.16b, v2.16b          // maxIdx A
	mov	v19.16b, v18.16b        // maxIdx B
	mov	v4.16b, v2.16b          // curIdx A
	mov	v20.16b, v18.16b        // curIdx B
	movi	v5.4s, #8
	add	x10, x0, #32            // element pointer
	add	x11, x0, x1, lsl #2     // row end
.Lmm_loop:
	cmp	x10, x11
	b.ge	.Lmm_merge
	ldp	q6, q21, [x10], #32
	add	v4.4s, v4.4s, v5.4s     // curIdx += 8
	add	v20.4s, v20.4s, v5.4s
	fcmgt	v7.4s, v0.4s, v6.4s     // v < minV (ordered, strict)
	bit	v0.16b, v6.16b, v7.16b
	bit	v2.16b, v4.16b, v7.16b
	fcmgt	v7.4s, v6.4s, v1.4s     // v > maxV
	bit	v1.16b, v6.16b, v7.16b
	bit	v3.16b, v4.16b, v7.16b
	fcmgt	v22.4s, v16.4s, v21.4s
	bit	v16.16b, v21.16b, v22.16b
	bit	v18.16b, v20.16b, v22.16b
	fcmgt	v22.4s, v21.4s, v17.4s
	bit	v17.16b, v21.16b, v22.16b
	bit	v19.16b, v20.16b, v22.16b
	b	.Lmm_loop
.Lmm_merge:
	// fold tracker B into A with the same lexicographic rule
	fcmgt	v8.4s, v0.4s, v16.4s
	fcmeq	v9.4s, v16.4s, v0.4s
	cmgt	v10.4s, v2.4s, v18.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v0.16b, v16.16b, v8.16b
	bit	v2.16b, v18.16b, v8.16b
	fcmgt	v8.4s, v17.4s, v1.4s
	fcmeq	v9.4s, v17.4s, v1.4s
	cmgt	v10.4s, v3.4s, v19.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v1.16b, v17.16b, v8.16b
	bit	v3.16b, v19.16b, v8.16b
.Lmm_reduce:
	// fold (value, index) lanes pairwise: take = (v'<v) | (v'==v & i'<i)
	ext	v6.16b, v0.16b, v0.16b, #8
	ext	v7.16b, v2.16b, v2.16b, #8
	fcmgt	v8.4s, v0.4s, v6.4s
	fcmeq	v9.4s, v6.4s, v0.4s
	cmgt	v10.4s, v2.4s, v7.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v0.16b, v6.16b, v8.16b
	bit	v2.16b, v7.16b, v8.16b
	rev64	v6.4s, v0.4s
	rev64	v7.4s, v2.4s
	fcmgt	v8.4s, v0.4s, v6.4s
	fcmeq	v9.4s, v6.4s, v0.4s
	cmgt	v10.4s, v2.4s, v7.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v0.16b, v6.16b, v8.16b
	bit	v2.16b, v7.16b, v8.16b
	ext	v6.16b, v1.16b, v1.16b, #8
	ext	v7.16b, v3.16b, v3.16b, #8
	fcmgt	v8.4s, v6.4s, v1.4s
	fcmeq	v9.4s, v6.4s, v1.4s
	cmgt	v10.4s, v3.4s, v7.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v1.16b, v6.16b, v8.16b
	bit	v3.16b, v7.16b, v8.16b
	rev64	v6.4s, v1.4s
	rev64	v7.4s, v3.4s
	fcmgt	v8.4s, v6.4s, v1.4s
	fcmeq	v9.4s, v6.4s, v1.4s
	cmgt	v10.4s, v3.4s, v7.4s
	and	v9.16b, v9.16b, v10.16b
	orr	v8.16b, v8.16b, v9.16b
	bit	v1.16b, v6.16b, v8.16b
	bit	v3.16b, v7.16b, v8.16b
	ret

// ---------------------------------------------------------------------------
// RGBAToGray: x0=dst, x1=len (multiple of 8), x2=src.
// gray = (4899*R + 9617*G + 1868*B + 8192) >> 14. Integer, exact in any
// association; umull sums stay far below 2^31.
.global RGBAToGray
RGBAToGray:
	mov	x9, #0x1323             // 4899
	movk	x9, #0x2591, lsl #16    // 9617
	movk	x9, #0x074c, lsl #32    // 1868 (alpha word 0)
	dup	v29.2d, x9              // coef u16 x8
	movi	v28.4s, #0x20, lsl #8   // 8192
	mov	x9, #0
	and	x10, x1, #-16           // end of 16-pixel blocks
	cmp	x9, x10
	b.ge	.Lgr_mid
.Lgr_loop:
	ldp	q1, q2, [x2], #32       // 16 RGBA pixels
	ldp	q23, q24, [x2], #32
	uxtl	v3.8h, v1.8b            // px 0,1 channels
	uxtl2	v4.8h, v1.16b           // px 2,3
	umull	v5.4s, v3.4h, v29.4h    // px0 weighted channels
	umull2	v6.4s, v3.8h, v29.8h    // px1
	umull	v7.4s, v4.4h, v29.4h    // px2
	umull2	v16.4s, v4.8h, v29.8h   // px3
	addp	v5.4s, v5.4s, v6.4s     // (rg0, b0, rg1, b1)
	addp	v7.4s, v7.4s, v16.4s    // (rg2, b2, rg3, b3)
	addp	v19.4s, v5.4s, v7.4s    // per-pixel sums 0..3
	uxtl	v3.8h, v2.8b
	uxtl2	v4.8h, v2.16b
	umull	v6.4s, v3.4h, v29.4h
	umull2	v16.4s, v3.8h, v29.8h
	umull	v7.4s, v4.4h, v29.4h
	umull2	v17.4s, v4.8h, v29.8h
	addp	v6.4s, v6.4s, v16.4s
	addp	v7.4s, v7.4s, v17.4s
	addp	v20.4s, v6.4s, v7.4s    // per-pixel sums 4..7
	uxtl	v3.8h, v23.8b
	uxtl2	v4.8h, v23.16b
	umull	v5.4s, v3.4h, v29.4h
	umull2	v6.4s, v3.8h, v29.8h
	umull	v7.4s, v4.4h, v29.4h
	umull2	v16.4s, v4.8h, v29.8h
	addp	v5.4s, v5.4s, v6.4s
	addp	v7.4s, v7.4s, v16.4s
	addp	v21.4s, v5.4s, v7.4s    // per-pixel sums 8..11
	uxtl	v3.8h, v24.8b
	uxtl2	v4.8h, v24.16b
	umull	v6.4s, v3.4h, v29.4h
	umull2	v16.4s, v3.8h, v29.8h
	umull	v7.4s, v4.4h, v29.4h
	umull2	v17.4s, v4.8h, v29.8h
	addp	v6.4s, v6.4s, v16.4s
	addp	v7.4s, v7.4s, v17.4s
	addp	v22.4s, v6.4s, v7.4s    // per-pixel sums 12..15
	add	v19.4s, v19.4s, v28.4s
	add	v20.4s, v20.4s, v28.4s
	add	v21.4s, v21.4s, v28.4s
	add	v22.4s, v22.4s, v28.4s
	ushr	v19.4s, v19.4s, #14
	ushr	v20.4s, v20.4s, #14
	ushr	v21.4s, v21.4s, #14
	ushr	v22.4s, v22.4s, #14
	xtn	v19.4h, v19.4s
	xtn2	v19.8h, v20.4s
	xtn	v19.8b, v19.8h
	xtn	v21.4h, v21.4s
	xtn2	v21.8h, v22.4s
	xtn	v21.8b, v21.8h
	mov	v19.d[1], v21.d[0]
	str	q19, [x0, x9]
	add	x9, x9, #16
	cmp	x9, x10
	b.lt	.Lgr_loop
.Lgr_mid:
	cmp	x9, x1
	b.ge	.Lgr_done
	ldp	q1, q2, [x2], #32       // final 8 pixels
	uxtl	v3.8h, v1.8b
	uxtl2	v4.8h, v1.16b
	umull	v5.4s, v3.4h, v29.4h
	umull2	v6.4s, v3.8h, v29.8h
	umull	v7.4s, v4.4h, v29.4h
	umull2	v16.4s, v4.8h, v29.8h
	addp	v5.4s, v5.4s, v6.4s
	addp	v7.4s, v7.4s, v16.4s
	addp	v5.4s, v5.4s, v7.4s
	uxtl	v3.8h, v2.8b
	uxtl2	v4.8h, v2.16b
	umull	v6.4s, v3.4h, v29.4h
	umull2	v16.4s, v3.8h, v29.8h
	umull	v7.4s, v4.4h, v29.4h
	umull2	v17.4s, v4.8h, v29.8h
	addp	v6.4s, v6.4s, v16.4s
	addp	v7.4s, v7.4s, v17.4s
	addp	v6.4s, v6.4s, v7.4s
	add	v5.4s, v5.4s, v28.4s
	add	v6.4s, v6.4s, v28.4s
	ushr	v5.4s, v5.4s, #14
	ushr	v6.4s, v6.4s, #14
	xtn	v5.4h, v5.4s
	xtn2	v5.8h, v6.4s
	xtn	v5.8b, v5.8h
	str	d5, [x0, x9]
.Lgr_done:
	ret

// ---------------------------------------------------------------------------
// SlideCols1: x0=colSum, x1=len (multiple of 8), x2=colSum2, x3=rsub,
// x4=radd. colSum[x] += a-b; colSum2[x] += a*a - b*b, computed as
// (a-b)*(a+b) — the same exact integer.
.global SlideCols1
SlideCols1:
	mov	x9, #0
	and	x10, x1, #-16           // end of 16-pixel blocks
	cmp	x9, x10
	b.ge	.Ls1_mid
.Ls1_loop:
	ldr	q0, [x4, x9]            // a bytes x16
	ldr	q25, [x3, x9]           // b bytes x16
	uxtl	v1.8h, v0.8b
	uxtl	v2.8h, v25.8b
	uxtl2	v26.8h, v0.16b
	uxtl2	v27.8h, v25.16b
	sub	v3.8h, v1.8h, v2.8h     // d in [-255, 255]
	add	v4.8h, v1.8h, v2.8h     // s in [0, 510]
	sub	v28.8h, v26.8h, v27.8h
	add	v29.8h, v26.8h, v27.8h
	sshll	v5.4s, v3.4h, #0
	sshll2	v6.4s, v3.8h, #0
	sshll	v30.4s, v28.4h, #0
	sshll2	v31.4s, v28.8h, #0
	ldp	q16, q17, [x0]
	add	v16.4s, v16.4s, v5.4s
	add	v17.4s, v17.4s, v6.4s
	stp	q16, q17, [x0], #32
	ldp	q16, q17, [x0]
	add	v16.4s, v16.4s, v30.4s
	add	v17.4s, v17.4s, v31.4s
	stp	q16, q17, [x0], #32
	smull	v7.4s, v3.4h, v4.4h     // a²-b² per px (|.| <= 65025)
	smull2	v18.4s, v3.8h, v4.8h
	smull	v5.4s, v28.4h, v29.4h
	smull2	v6.4s, v28.8h, v29.8h
	sshll	v19.2d, v7.2s, #0
	sshll2	v20.2d, v7.4s, #0
	sshll	v21.2d, v18.2s, #0
	sshll2	v22.2d, v18.4s, #0
	sshll	v26.2d, v5.2s, #0
	sshll2	v27.2d, v5.4s, #0
	sshll	v30.2d, v6.2s, #0
	sshll2	v31.2d, v6.4s, #0
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v19.2d
	add	v24.2d, v24.2d, v20.2d
	stp	q23, q24, [x2], #32
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v21.2d
	add	v24.2d, v24.2d, v22.2d
	stp	q23, q24, [x2], #32
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v26.2d
	add	v24.2d, v24.2d, v27.2d
	stp	q23, q24, [x2], #32
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v30.2d
	add	v24.2d, v24.2d, v31.2d
	stp	q23, q24, [x2], #32
	add	x9, x9, #16
	cmp	x9, x10
	b.lt	.Ls1_loop
.Ls1_mid:
	cmp	x9, x1
	b.ge	.Ls1_done
	ldr	d1, [x4, x9]            // final 8 pixels
	ldr	d2, [x3, x9]
	uxtl	v1.8h, v1.8b
	uxtl	v2.8h, v2.8b
	sub	v3.8h, v1.8h, v2.8h
	add	v4.8h, v1.8h, v2.8h
	sshll	v5.4s, v3.4h, #0
	sshll2	v6.4s, v3.8h, #0
	ldp	q16, q17, [x0]
	add	v16.4s, v16.4s, v5.4s
	add	v17.4s, v17.4s, v6.4s
	stp	q16, q17, [x0], #32
	smull	v7.4s, v3.4h, v4.4h
	smull2	v18.4s, v3.8h, v4.8h
	sshll	v19.2d, v7.2s, #0
	sshll2	v20.2d, v7.4s, #0
	sshll	v21.2d, v18.2s, #0
	sshll2	v22.2d, v18.4s, #0
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v19.2d
	add	v24.2d, v24.2d, v20.2d
	stp	q23, q24, [x2], #32
	ldp	q23, q24, [x2]
	add	v23.2d, v23.2d, v21.2d
	add	v24.2d, v24.2d, v22.2d
	stp	q23, q24, [x2], #32
.Ls1_done:
	ret

// ---------------------------------------------------------------------------
// SlideCols4: x0=colSum, x1=colSum2, x2=pixels (multiple of 8), x3=rsub,
// x4=radd, x5=cn. All four colSum lanes slide per pixel; colSum2 gets the
// summed squared channel deltas with alpha masked out when cn == 3.
.global SlideCols4
SlideCols4:
	movi	v30.2d, #0xffffffffffffffff
	cmp	x5, #3
	b.ne	.Ls4_start
	mov	x9, #0x0000ffffffffffff
	dup	v30.2d, x9              // zero the alpha words
.Ls4_start:
	cbz	x2, .Ls4_done
	mov	x9, #0
.Ls4_loop:
	ldp	q1, q2, [x4], #32       // a: 8 px
	ldp	q3, q4, [x3], #32       // b
	// colSum += a-b over 32 int32 lanes
	uxtl	v5.8h, v1.8b
	uxtl2	v6.8h, v1.16b
	uxtl	v7.8h, v3.8b
	uxtl2	v16.8h, v3.16b
	sub	v5.8h, v5.8h, v7.8h     // d px0,1
	sub	v6.8h, v6.8h, v16.8h    // d px2,3
	uxtl	v17.8h, v2.8b
	uxtl2	v18.8h, v2.16b
	uxtl	v19.8h, v4.8b
	uxtl2	v20.8h, v4.16b
	sub	v17.8h, v17.8h, v19.8h
	sub	v18.8h, v18.8h, v20.8h
	sshll	v21.4s, v5.4h, #0
	sshll2	v22.4s, v5.8h, #0
	sshll	v23.4s, v6.4h, #0
	sshll2	v24.4s, v6.8h, #0
	ldp	q25, q26, [x0]
	add	v25.4s, v25.4s, v21.4s
	add	v26.4s, v26.4s, v22.4s
	stp	q25, q26, [x0], #32
	ldp	q25, q26, [x0]
	add	v25.4s, v25.4s, v23.4s
	add	v26.4s, v26.4s, v24.4s
	stp	q25, q26, [x0], #32
	sshll	v21.4s, v17.4h, #0
	sshll2	v22.4s, v17.8h, #0
	sshll	v23.4s, v18.4h, #0
	sshll2	v24.4s, v18.8h, #0
	ldp	q25, q26, [x0]
	add	v25.4s, v25.4s, v21.4s
	add	v26.4s, v26.4s, v22.4s
	stp	q25, q26, [x0], #32
	ldp	q25, q26, [x0]
	add	v25.4s, v25.4s, v23.4s
	add	v26.4s, v26.4s, v24.4s
	stp	q25, q26, [x0], #32
	// colSum2 += per-pixel sum of masked (a-b)*(a+b)
	uxtl	v7.8h, v3.8b            // rebuild b16 (px0-3 pair 1)
	uxtl2	v16.8h, v3.16b
	uxtl	v19.8h, v1.8b           // a16
	uxtl2	v20.8h, v1.16b
	add	v21.8h, v19.8h, v7.8h   // s px0,1
	add	v22.8h, v20.8h, v16.8h  // s px2,3
	and	v5.16b, v5.16b, v30.16b // masked d
	and	v6.16b, v6.16b, v30.16b
	and	v21.16b, v21.16b, v30.16b
	and	v22.16b, v22.16b, v30.16b
	smull	v23.4s, v5.4h, v21.4h   // px0 channel products
	smull2	v24.4s, v5.8h, v21.8h   // px1
	smull	v25.4s, v6.4h, v22.4h   // px2
	smull2	v26.4s, v6.8h, v22.8h   // px3
	addp	v23.4s, v23.4s, v24.4s
	addp	v25.4s, v25.4s, v26.4s
	addp	v23.4s, v23.4s, v25.4s  // per-px sums 0..3
	uxtl	v7.8h, v4.8b            // b16 px4-7
	uxtl2	v16.8h, v4.16b
	uxtl	v19.8h, v2.8b
	uxtl2	v20.8h, v2.16b
	add	v21.8h, v19.8h, v7.8h
	add	v22.8h, v20.8h, v16.8h
	and	v17.16b, v17.16b, v30.16b
	and	v18.16b, v18.16b, v30.16b
	and	v21.16b, v21.16b, v30.16b
	and	v22.16b, v22.16b, v30.16b
	smull	v24.4s, v17.4h, v21.4h
	smull2	v25.4s, v17.8h, v21.8h
	smull	v26.4s, v18.4h, v22.4h
	smull2	v27.4s, v18.8h, v22.8h
	addp	v24.4s, v24.4s, v25.4s
	addp	v26.4s, v26.4s, v27.4s
	addp	v24.4s, v24.4s, v26.4s  // per-px sums 4..7
	sshll	v25.2d, v23.2s, #0
	sshll2	v26.2d, v23.4s, #0
	sshll	v27.2d, v24.2s, #0
	sshll2	v28.2d, v24.4s, #0
	ldp	q16, q17, [x1]
	add	v16.2d, v16.2d, v25.2d
	add	v17.2d, v17.2d, v26.2d
	stp	q16, q17, [x1], #32
	ldp	q16, q17, [x1]
	add	v16.2d, v16.2d, v27.2d
	add	v17.2d, v17.2d, v28.2d
	stp	q16, q17, [x1], #32
	add	x9, x9, #8
	cmp	x9, x2
	b.lt	.Ls4_loop
.Ls4_done:
	ret

// ---------------------------------------------------------------------------
// SpillStats1: x0=wt(s0 lane), x1=n (multiple of 4, > 0), x2=stride,
// x3=lo, x4=hi, x5=lo2, x6=hi2, x7=s0 (in/out), x8=s2 (in/out),
// x9=area. The cn=1 normalize spill, 4 elements per pass, mirroring the
// amd64 kernel: window sums stay below 2^31 (caller-gated area), so
// every wide product is one smull 32x32->64, the running idiff advances
// by Didiff = area*d2 - 2*d*s0 - d*d, and int64 prefix regrouping is
// exact. Emitted lanes are float32(float64(v)): scvtf.2d is the one
// correct rounding to f64, fcvtn the one to f32 — the scalar spill's
// exact sequence.
// Register map: x11=idiff base x12=s2 base x13/x15=scalar scratch x14=i
// v28=area.4s v31=0; per pass: v0=d32 v4=s0v32 v5/v6=s0 widen
// v7/v8=d2 v9/v10=inclusive prefixes v12/v13=t2v v15=idiff0
// v16=d2_32 v17-v22=products v23/v24=idiff64
.global SpillStats1
SpillStats1:
	lsl	x13, x2, #2             // stride in bytes
	add	x11, x0, x13            // idiff lane base
	add	x12, x11, x13           // s2 lane base
	dup	v28.4s, w9              // area (< 2^23)
	movi	v31.16b, #0
	mov	x14, #0                 // i
.Lsp1_loop:
	// idiff0 = area*s2 - s0*s0 at quad entry (scalar, exact)
	mul	x13, x8, x9
	mul	x15, x7, x7
	sub	x13, x13, x15
	dup	v15.2d, x13             // idiff0
	// d = hi-lo (int32x4); 32-bit exclusive prefix -> s0v
	add	x15, x4, x14, lsl #2
	ldr	q0, [x15]
	add	x15, x3, x14, lsl #2
	ldr	q1, [x15]
	sub	v0.4s, v0.4s, v1.4s     // d
	ext	v1.16b, v31.16b, v0.16b, #12
	add	v2.4s, v0.4s, v1.4s
	ext	v3.16b, v31.16b, v2.16b, #8
	add	v2.4s, v2.4s, v3.4s     // inclusive prefix of d
	ext	v3.16b, v31.16b, v2.16b, #12
	dup	v4.4s, w7
	add	v4.4s, v4.4s, v3.4s     // s0v (pre-slide, < 2^31)
	smov	x15, v2.s[3]
	add	x7, x7, x15             // s0 advanced past the quad
	// s0 lane: widen and convert (one rounding each step)
	ushll	v5.2d, v4.2s, #0
	ushll2	v6.2d, v4.4s, #0
	scvtf	v5.2d, v5.2d
	scvtf	v6.2d, v6.2d
	fcvtn	v5.2s, v5.2d
	fcvtn2	v5.4s, v6.2d
	add	x15, x0, x14, lsl #2
	str	q5, [x15]
	// d2 = hi2-lo2 (int64x2 x2)
	add	x15, x6, x14, lsl #3
	ldp	q7, q8, [x15]
	add	x15, x5, x14, lsl #3
	ldp	q9, q10, [x15]
	sub	v7.2d, v7.2d, v9.2d
	sub	v8.2d, v8.2d, v10.2d
	// t2v = s2 + 64-bit exclusive prefix of d2; advance scalar s2
	ext	v9.16b, v31.16b, v7.16b, #8
	add	v9.2d, v9.2d, v7.2d     // incl A
	ext	v10.16b, v31.16b, v8.16b, #8
	add	v10.2d, v10.2d, v8.2d
	dup	v11.2d, v9.d[1]
	add	v10.2d, v10.2d, v11.2d  // incl B
	ext	v12.16b, v31.16b, v9.16b, #8
	ext	v13.16b, v9.16b, v10.16b, #8
	dup	v14.2d, x8
	add	v12.2d, v12.2d, v14.2d  // t2v lanes 0-1
	add	v13.2d, v13.2d, v14.2d  // t2v lanes 2-3
	umov	x15, v10.d[1]
	add	x8, x8, x15             // s2 advanced past the quad
	// Didiff = area*d2 - 2*d*s0v - d*d (each product 32x32->64)
	xtn	v16.2s, v7.2d
	xtn2	v16.4s, v8.2d           // d2 as int32 (|d2| < 2^31: th gate)
	smull	v17.2d, v16.2s, v28.2s
	smull2	v18.2d, v16.4s, v28.4s // area*d2
	smull	v19.2d, v0.2s, v4.2s
	smull2	v20.2d, v0.4s, v4.4s   // d*s0v
	shl	v19.2d, v19.2d, #1
	shl	v20.2d, v20.2d, #1
	sub	v17.2d, v17.2d, v19.2d
	sub	v18.2d, v18.2d, v20.2d
	smull	v21.2d, v0.2s, v0.2s
	smull2	v22.2d, v0.4s, v0.4s   // d*d
	sub	v17.2d, v17.2d, v21.2d  // Didiff lanes 0-1
	sub	v18.2d, v18.2d, v22.2d  // Didiff lanes 2-3
	// idiff64 = idiff0 + 64-bit exclusive prefix of Didiff
	ext	v9.16b, v31.16b, v17.16b, #8
	add	v9.2d, v9.2d, v17.2d
	ext	v10.16b, v31.16b, v18.16b, #8
	add	v10.2d, v10.2d, v18.2d
	dup	v11.2d, v9.d[1]
	add	v10.2d, v10.2d, v11.2d
	ext	v23.16b, v31.16b, v9.16b, #8
	ext	v24.16b, v9.16b, v10.16b, #8
	add	v23.2d, v23.2d, v15.2d  // idiff64 lanes 0-1
	add	v24.2d, v24.2d, v15.2d  // idiff64 lanes 2-3
	// converts: scvtf (one rounding to f64), fcvtn (one to f32); stores
	scvtf	v23.2d, v23.2d
	scvtf	v24.2d, v24.2d
	fcvtn	v23.2s, v23.2d
	fcvtn2	v23.4s, v24.2d
	add	x15, x11, x14, lsl #2
	str	q23, [x15]              // idiff lane
	scvtf	v12.2d, v12.2d
	scvtf	v13.2d, v13.2d
	fcvtn	v12.2s, v12.2d
	fcvtn2	v12.4s, v13.2d
	add	x15, x12, x14, lsl #2
	str	q12, [x15]              // s2 lane
	add	x14, x14, #4
	cmp	x14, x1
	b.lt	.Lsp1_loop
	ret
