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
// NormRow: x0=rrow, x1=crow, x2=wt, x3=stride, x4=n (multiple of 4, > 0),
// x5=cn, x6=mean; v0.d[0]=invArea, v1.d[0]=eps, v2.d[0]=templNorm.
// The TM_CCOEFF_NORMED tail at f64x2, mirroring the amd64 norm_tail4 ops:
// clamp/lim/sqrt/guarded-div/copysign-sat, all exact or correctly rounded.
.global NormRow
NormRow:
	dup	v28.2d, v0.d[0]         // invArea
	dup	v29.2d, v1.d[0]         // eps
	dup	v30.2d, v2.d[0]         // templNorm
	fmov	v27.2d, #0.5
	fmov	v25.2d, #1.0
	fmov	v24.2d, #1.125
	mov	x9, #0x8000000000000000
	dup	v23.2d, x9              // f64 sign mask
	movi	v22.2d, #0              // +0.0
	lsl	x7, x3, #3              // stride in bytes
	mov	x8, #0                  // i
	// n is a multiple of 4: each pass handles two f64x2 halves and the
	// k-loop shares one mean load and one 32-byte wt load across both.
.Lnr_loop:
	ldr	q3, [x1], #16           // crow[i..i+3] (4 f32)
	fcvtl2	v13.2d, v3.4s           // num, high half (exact widen)
	fcvtl	v3.2d, v3.2s            // num, low half
	movi	v4.2d, #0               // wndMean2 accumulators
	movi	v14.2d, #0
	add	x9, x2, x8, lsl #3      // &wt[i]
	mov	x10, x5                 // k = cn
	mov	x11, x6                 // &mean[0]
.Lnr_k:
	ldp	q5, q15, [x9]           // t lanes, both halves
	add	x9, x9, x7
	ldr	d7, [x11], #8
	dup	v7.2d, v7.d[0]          // mean[k]
	fmul	v6.2d, v5.2d, v5.2d
	fadd	v4.2d, v4.2d, v6.2d     // wndMean2 += t*t
	fmul	v16.2d, v15.2d, v15.2d
	fadd	v14.2d, v14.2d, v16.2d
	fmul	v6.2d, v5.2d, v7.2d
	fsub	v3.2d, v3.2d, v6.2d     // num -= t*mean[k]
	fmul	v16.2d, v15.2d, v7.2d
	fsub	v13.2d, v13.2d, v16.2d
	subs	x10, x10, #1
	b.ne	.Lnr_k
	ldp	q8, q18, [x9]           // s2d = q2 lanes (wt + cn*stride)
	fmul	v4.2d, v4.2d, v28.2d    // wndMean2 *= invArea
	fmul	v14.2d, v14.2d, v28.2d
	fsub	v5.2d, v8.2d, v4.2d     // diff2
	fsub	v15.2d, v18.2d, v14.2d
	fmax	v5.2d, v5.2d, v22.2d    // clamp at 0
	fmax	v15.2d, v15.2d, v22.2d
	fmul	v6.2d, v8.2d, v29.2d    // lim = eps*s2d
	fmul	v16.2d, v18.2d, v29.2d
	fmin	v6.2d, v6.2d, v27.2d    // min(lim, 0.5)
	fmin	v16.2d, v16.2d, v27.2d
	fsqrt	v7.2d, v5.2d
	fsqrt	v17.2d, v15.2d
	fmul	v7.2d, v7.2d, v30.2d    // den = sqrt(diff2)*templNorm
	fmul	v17.2d, v17.2d, v30.2d
	fcmgt	v9.2d, v5.2d, v6.2d     // diff2 > lim
	fcmgt	v19.2d, v15.2d, v16.2d
	and	v7.16b, v7.16b, v9.16b  // den, or 0 below the guard
	and	v17.16b, v17.16b, v19.16b
	fabs	v10.2d, v3.2d           // an = |num| (bit op, exact)
	fabs	v20.2d, v13.2d
	fcmgt	v11.2d, v7.2d, v10.2d   // m1 = an < den
	fcmgt	v21.2d, v17.2d, v20.2d
	mov	v12.16b, v11.16b
	bsl	v12.16b, v7.16b, v25.16b // divisor = m1 ? den : 1.0
	and	v5.16b, v3.16b, v11.16b // m1 ? num : 0
	fdiv	v5.2d, v5.2d, v12.2d    // dv
	mov	v12.16b, v21.16b
	bsl	v12.16b, v17.16b, v25.16b
	and	v15.16b, v13.16b, v21.16b
	fdiv	v15.2d, v15.2d, v12.2d
	fmul	v6.2d, v7.2d, v24.2d    // den*1.125
	fcmgt	v6.2d, v6.2d, v10.2d    // m2 = an < den*1.125
	fmul	v16.2d, v17.2d, v24.2d
	fcmgt	v16.2d, v16.2d, v20.2d
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
	fcvtn	v9.2s, v9.2d            // correctly rounded narrow
	fcvtn2	v9.4s, v19.2d
	str	q9, [x0], #16
	add	x8, x8, #4
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
// SlideSpill1: x0=wt, x1=n, x2=q2, x3=lo, x4=hi, x5=lo2, x6=hi2,
// x7=s0, x8=s2. Leaves the advanced sums in x7/x8 for the template to
// store. wt[i]=float64(s0), q2[i]=float64(s2), then s0 += hi[i]-lo[i],
// s2 += hi2[i]-lo2[i]. scvtf on integers < 2^52 is exact, and the sum
// chains run scalar in registers — identical values to the Go loop.
.global SlideSpill1
SlideSpill1:
	and	x12, x1, #-2
	mov	x13, #0
.Lss_loop:
	cmp	x13, x12
	b.ge	.Lss_tail
	ldrsw	x9, [x4], #4            // hi[i]
	ldrsw	x10, [x3], #4           // lo[i]
	sub	x9, x9, x10
	add	x9, x7, x9              // s1 = s0 + d0
	fmov	d0, x7
	mov	v0.d[1], x9
	scvtf	v0.2d, v0.2d            // exact
	str	q0, [x0], #16
	ldrsw	x10, [x4], #4
	ldrsw	x11, [x3], #4
	sub	x10, x10, x11
	add	x7, x9, x10             // s0 advanced past both
	ldr	x9, [x6], #8            // hi2[i]
	ldr	x10, [x5], #8           // lo2[i]
	sub	x9, x9, x10
	add	x9, x8, x9              // t1 = s2 + e0
	fmov	d1, x8
	mov	v1.d[1], x9
	scvtf	v1.2d, v1.2d
	str	q1, [x2], #16
	ldr	x10, [x6], #8
	ldr	x11, [x5], #8
	sub	x10, x10, x11
	add	x8, x9, x10             // s2 advanced
	add	x13, x13, #2
	b	.Lss_loop
.Lss_tail:
	cmp	x13, x1
	b.ge	.Lss_done
	scvtf	d0, x7
	str	d0, [x0], #8
	scvtf	d1, x8
	str	d1, [x2], #8
	ldrsw	x9, [x4], #4
	ldrsw	x10, [x3], #4
	sub	x9, x9, x10
	add	x7, x7, x9
	ldr	x9, [x6], #8
	ldr	x10, [x5], #8
	sub	x9, x9, x10
	add	x8, x8, x9
	add	x13, x13, #1
	b	.Lss_tail
.Lss_done:
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
