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

.text

// ---------------------------------------------------------------------------
// FFTStages: x0=a_base, x1=n (power of two >= 8), x2=tw_base, x3=inverse(0/1)
// Runs the full radix-2 cascade over bit-reversed data. Stages half=1,2 are
// fused into one pass (every 4-complex group is closed under both stages),
// then the generic cascade runs from half=4 in 2-complex steps.
.global FFTStages
FFTStages:
	lsl	w9, w3, #31
	dup	v29.4s, w9              // s: 0x80000000 lanes when inverse
	mov	w9, #0x80000000
	dup	v26.2d, x9              // evensign: flip even 32-bit lanes
	movz	x9, #0x8000, lsl #48
	movk	x9, #0x8000, lsl #16
	movi	v25.2d, #0
	mov	v25.d[1], x9            // highsign: flip the high complex only
	ldr	d30, [x2, #8]           // tw[1]
	dup	v27.4s, v30.s[0]        // wr1
	dup	v28.4s, v30.s[1]
	eor	v28.16b, v28.16b, v29.16b // wi1 = s*imag(tw[1])
	ldr	q30, [x2, #16]          // tw[2], tw[3]
	trn1	v23.4s, v30.4s, v30.4s  // (w2r, w2r, w3r, w3r)
	trn2	v24.4s, v30.4s, v30.4s
	eor	v24.16b, v24.16b, v29.16b // s*imag
	mov	x9, x0
	mov	x10, x1                 // complexes remaining

.Lfs_s12:
	ldp	q3, q4, [x9]            // c0,c1 | c2,c3
	// half=1 on (c0,c1): butterfly with tw[1]; the p±qw pair is built as
	// (c0,c0) + (qw,-qw) — negation by xor is exact, x-y == x+(-y)
	rev64	v5.4s, v3.4s
	fmul	v6.4s, v3.4s, v27.4s
	fmul	v7.4s, v5.4s, v28.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // q*w in the odd complex lane
	dup	v8.2d, v6.d[1]          // (qw, qw)
	eor	v8.16b, v8.16b, v25.16b // (qw, -qw)
	dup	v9.2d, v3.d[0]          // (p, p)
	fadd	v3.4s, v9.4s, v8.4s     // (p+qw, p-qw)
	// half=1 on (c2,c3)
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v27.4s
	fmul	v7.4s, v5.4s, v28.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	dup	v8.2d, v6.d[1]
	eor	v8.16b, v8.16b, v25.16b
	dup	v9.2d, v4.d[0]
	fadd	v4.4s, v9.4s, v8.4s
	// half=2 across the pair: (c0,c2) w=tw[2], (c1,c3) w=tw[3]
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v23.4s
	fmul	v7.4s, v5.4s, v24.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	fadd	v10.4s, v3.4s, v6.4s
	fsub	v11.4s, v3.4s, v6.4s
	stp	q10, q11, [x9], #32
	subs	x10, x10, #4
	b.ne	.Lfs_s12

	mov	x4, #4                  // half
.Lfs_half:
	cmp	x4, x1
	b.ge	.Lfs_done
	add	x5, x2, x4, lsl #3      // w = tw + half
	mov	x6, #0                  // i
.Lfs_i:
	add	x7, x0, x6, lsl #3      // p
	add	x8, x7, x4, lsl #3      // q = p + half
	mov	x9, x5                  // wj
	mov	x10, x4                 // j countdown, steps of 4
.Lfs_j:
	ldp	q1, q10, [x8]           // two q pairs
	ldp	q2, q12, [x9], #32      // two twiddle pairs
	trn1	v3.4s, v2.4s, v2.4s     // wr lanes
	trn2	v4.4s, v2.4s, v2.4s     // wi lanes
	eor	v4.16b, v4.16b, v29.16b
	rev64	v5.4s, v1.4s
	fmul	v6.4s, v1.4s, v3.4s
	fmul	v7.4s, v5.4s, v4.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // q*w
	trn1	v13.4s, v12.4s, v12.4s
	trn2	v14.4s, v12.4s, v12.4s
	eor	v14.16b, v14.16b, v29.16b
	rev64	v15.4s, v10.4s
	fmul	v16.4s, v10.4s, v13.4s
	fmul	v17.4s, v15.4s, v14.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	ldp	q0, q11, [x7]
	fadd	v8.4s, v0.4s, v6.4s
	fsub	v9.4s, v0.4s, v6.4s
	fadd	v18.4s, v11.4s, v16.4s
	fsub	v19.4s, v11.4s, v16.4s
	stp	q8, q18, [x7], #32
	stp	q9, q19, [x8], #32
	subs	x10, x10, #4
	b.ne	.Lfs_j
	add	x6, x6, x4, lsl #1      // i += half*2
	cmp	x6, x1
	b.lt	.Lfs_i
	lsl	x4, x4, #1
	b	.Lfs_half
.Lfs_done:
	ret

// ---------------------------------------------------------------------------
// FFTColsBfly: x0=p_base, x1=len, x2=q_base; v0.s[0]=wr, v1.s[0]=wi
// (twiddle already sign-adjusted by the caller). p,q = p+q*w, p-q*w.
.global FFTColsBfly
FFTColsBfly:
	dup	v27.4s, v0.s[0]
	dup	v28.4s, v1.s[0]
	mov	w9, #0x80000000
	dup	v26.2d, x9              // evensign
	and	x3, x1, #-4
	add	x9, x0, x3, lsl #3      // p end of 4-complex blocks
	and	x4, x1, #-2
	add	x13, x0, x4, lsl #3     // p end of 2-complex blocks
	add	x10, x0, x1, lsl #3     // p true end
	mov	x11, x0
	mov	x12, x2
.Lcb_main:
	cmp	x11, x9
	b.ge	.Lcb_mid
	ldp	q1, q10, [x12]
	rev64	v5.4s, v1.4s
	fmul	v6.4s, v1.4s, v27.4s
	fmul	v7.4s, v5.4s, v28.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	rev64	v15.4s, v10.4s
	fmul	v16.4s, v10.4s, v27.4s
	fmul	v17.4s, v15.4s, v28.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	ldp	q0, q11, [x11]
	fadd	v8.4s, v0.4s, v6.4s
	fsub	v9.4s, v0.4s, v6.4s
	fadd	v18.4s, v11.4s, v16.4s
	fsub	v19.4s, v11.4s, v16.4s
	stp	q8, q18, [x11], #32
	stp	q9, q19, [x12], #32
	b	.Lcb_main
.Lcb_mid:
	cmp	x11, x13
	b.ge	.Lcb_tail
	ldr	q1, [x12]
	rev64	v5.4s, v1.4s
	fmul	v6.4s, v1.4s, v27.4s
	fmul	v7.4s, v5.4s, v28.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	ldr	q0, [x11]
	fadd	v8.4s, v0.4s, v6.4s
	fsub	v9.4s, v0.4s, v6.4s
	str	q8, [x11], #16
	str	q9, [x12], #16
.Lcb_tail:
	cmp	x11, x10
	b.ge	.Lcb_done
	ldr	d1, [x12]
	rev64	v5.2s, v1.2s
	fmul	v6.2s, v1.2s, v27.2s
	fmul	v7.2s, v5.2s, v28.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	ldr	d0, [x11]
	fadd	v8.2s, v0.2s, v6.2s
	fsub	v9.2s, v0.2s, v6.2s
	str	d8, [x11], #8
	str	d9, [x12], #8
	b	.Lcb_tail
.Lcb_done:
	ret

// ---------------------------------------------------------------------------
// FFTCols4: x0..x3 = r0..r3 bases, x4 = len;
// v0.s[0]/v1.s[0] = w1 re/im, v2.s[0]/v3.s[0] = w2a, v4.s[0]/v5.s[0] = w2b.
// Two fused column stages on a closed row quad, exact FFTColsBfly sequence
// per butterfly, values held in registers between the stages.
.global FFTCols4
FFTCols4:
	dup	v20.4s, v0.s[0]         // w1r
	dup	v21.4s, v1.s[0]         // w1i
	dup	v22.4s, v2.s[0]         // w2ar
	dup	v23.4s, v3.s[0]         // w2ai
	dup	v24.4s, v4.s[0]         // w2br
	dup	v25.4s, v5.s[0]         // w2bi
	mov	w9, #0x80000000
	dup	v26.2d, x9
	and	x5, x4, #-4
	add	x6, x0, x5, lsl #3      // r0 end of 4-complex blocks
	and	x8, x4, #-2
	add	x10, x0, x8, lsl #3     // r0 end of 2-complex blocks
	add	x7, x0, x4, lsl #3      // r0 true end
.Lc4_main:
	cmp	x0, x6
	b.ge	.Lc4_mid
	ldp	q1, q10, [x0]           // A
	ldp	q2, q11, [x1]           // B
	ldp	q3, q13, [x2]           // C
	ldp	q4, q14, [x3]           // D
	// stage 1: A,B = A±B*w1 ; C,D = C±D*w1  (two blocks interleaved)
	rev64	v5.4s, v2.4s
	fmul	v6.4s, v2.4s, v20.4s
	fmul	v7.4s, v5.4s, v21.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // B*w1
	fadd	v8.4s, v1.4s, v6.4s     // A'
	fsub	v2.4s, v1.4s, v6.4s     // B'
	rev64	v15.4s, v11.4s
	fmul	v16.4s, v11.4s, v20.4s
	fmul	v17.4s, v15.4s, v21.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	fadd	v18.4s, v10.4s, v16.4s  // A2'
	fsub	v11.4s, v10.4s, v16.4s  // B2'
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v20.4s
	fmul	v7.4s, v5.4s, v21.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // D*w1
	fadd	v9.4s, v3.4s, v6.4s     // C'
	fsub	v4.4s, v3.4s, v6.4s     // D'
	rev64	v15.4s, v14.4s
	fmul	v16.4s, v14.4s, v20.4s
	fmul	v17.4s, v15.4s, v21.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	fadd	v19.4s, v13.4s, v16.4s  // C2'
	fsub	v14.4s, v13.4s, v16.4s  // D2'
	// stage 2: A,C = A'±C'*w2a ; B,D = B'±D'*w2b
	rev64	v5.4s, v9.4s
	fmul	v6.4s, v9.4s, v22.4s
	fmul	v7.4s, v5.4s, v23.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // C'*w2a
	fadd	v1.4s, v8.4s, v6.4s     // r0 out
	fsub	v3.4s, v8.4s, v6.4s     // r2 out
	rev64	v15.4s, v19.4s
	fmul	v16.4s, v19.4s, v22.4s
	fmul	v17.4s, v15.4s, v23.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	fadd	v10.4s, v18.4s, v16.4s  // r0 out, block 2
	fsub	v13.4s, v18.4s, v16.4s  // r2 out, block 2
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v24.4s
	fmul	v7.4s, v5.4s, v25.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // D'*w2b
	fadd	v8.4s, v2.4s, v6.4s     // r1 out
	fsub	v4.4s, v2.4s, v6.4s     // r3 out
	rev64	v15.4s, v14.4s
	fmul	v16.4s, v14.4s, v24.4s
	fmul	v17.4s, v15.4s, v25.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	fadd	v18.4s, v11.4s, v16.4s  // r1 out, block 2
	fsub	v14.4s, v11.4s, v16.4s  // r3 out, block 2
	stp	q1, q10, [x0], #32
	stp	q8, q18, [x1], #32
	stp	q3, q13, [x2], #32
	stp	q4, q14, [x3], #32
	b	.Lc4_main
.Lc4_mid:
	cmp	x0, x10
	b.ge	.Lc4_tail
	ldr	q1, [x0]                // A
	ldr	q2, [x1]                // B
	ldr	q3, [x2]                // C
	ldr	q4, [x3]                // D
	rev64	v5.4s, v2.4s
	fmul	v6.4s, v2.4s, v20.4s
	fmul	v7.4s, v5.4s, v21.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // B*w1
	fadd	v8.4s, v1.4s, v6.4s     // A'
	fsub	v2.4s, v1.4s, v6.4s     // B'
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v20.4s
	fmul	v7.4s, v5.4s, v21.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // D*w1
	fadd	v9.4s, v3.4s, v6.4s     // C'
	fsub	v4.4s, v3.4s, v6.4s     // D'
	rev64	v5.4s, v9.4s
	fmul	v6.4s, v9.4s, v22.4s
	fmul	v7.4s, v5.4s, v23.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // C'*w2a
	fadd	v1.4s, v8.4s, v6.4s     // r0 out
	fsub	v3.4s, v8.4s, v6.4s     // r2 out
	rev64	v5.4s, v4.4s
	fmul	v6.4s, v4.4s, v24.4s
	fmul	v7.4s, v5.4s, v25.4s
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s     // D'*w2b
	fadd	v8.4s, v2.4s, v6.4s     // r1 out
	fsub	v4.4s, v2.4s, v6.4s     // r3 out
	str	q1, [x0], #16
	str	q8, [x1], #16
	str	q3, [x2], #16
	str	q4, [x3], #16
.Lc4_tail:
	cmp	x0, x7
	b.ge	.Lc4_done
	ldr	d1, [x0]
	ldr	d2, [x1]
	ldr	d3, [x2]
	ldr	d4, [x3]
	rev64	v5.2s, v2.2s
	fmul	v6.2s, v2.2s, v20.2s
	fmul	v7.2s, v5.2s, v21.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	fadd	v8.2s, v1.2s, v6.2s
	fsub	v2.2s, v1.2s, v6.2s
	rev64	v5.2s, v4.2s
	fmul	v6.2s, v4.2s, v20.2s
	fmul	v7.2s, v5.2s, v21.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	fadd	v9.2s, v3.2s, v6.2s
	fsub	v4.2s, v3.2s, v6.2s
	rev64	v5.2s, v9.2s
	fmul	v6.2s, v9.2s, v22.2s
	fmul	v7.2s, v5.2s, v23.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	fadd	v1.2s, v8.2s, v6.2s
	fsub	v3.2s, v8.2s, v6.2s
	rev64	v5.2s, v4.2s
	fmul	v6.2s, v4.2s, v24.2s
	fmul	v7.2s, v5.2s, v25.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	fadd	v8.2s, v2.2s, v6.2s
	fsub	v4.2s, v2.2s, v6.2s
	str	d1, [x0], #8
	str	d8, [x1], #8
	str	d3, [x2], #8
	str	d4, [x3], #8
	b	.Lc4_tail
.Lc4_done:
	ret

// ---------------------------------------------------------------------------
// MulConj: x0=spec_base, x1=len, x2=tspec_base. spec *= conj(tspec):
// (ar*br + ai*bi, ai*br - ar*bi). t1 = a*br, t2 = rev64(a)*bi, then
// (t1.e+t2.e, t1.o-t2.o) = t1 + (t2 ^ oddsign).
.global MulConj
MulConj:
	mov	x9, #0x8000000000000000
	dup	v26.2d, x9              // oddsign: flip odd 32-bit lanes
	and	x3, x1, #-4
	add	x9, x0, x3, lsl #3      // end of 4-complex blocks
	and	x4, x1, #-2
	add	x13, x0, x4, lsl #3     // end of 2-complex blocks
	add	x10, x0, x1, lsl #3
	mov	x11, x0
	mov	x12, x2
.Lmc_main:
	cmp	x11, x9
	b.ge	.Lmc_mid
	ldp	q1, q10, [x11]          // a
	ldp	q2, q12, [x12], #32     // b
	trn1	v3.4s, v2.4s, v2.4s     // br
	trn2	v4.4s, v2.4s, v2.4s     // bi
	rev64	v5.4s, v1.4s
	fmul	v6.4s, v1.4s, v3.4s     // t1
	fmul	v7.4s, v5.4s, v4.4s     // t2
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	trn1	v13.4s, v12.4s, v12.4s
	trn2	v14.4s, v12.4s, v12.4s
	rev64	v15.4s, v10.4s
	fmul	v16.4s, v10.4s, v13.4s
	fmul	v17.4s, v15.4s, v14.4s
	eor	v17.16b, v17.16b, v26.16b
	fadd	v16.4s, v16.4s, v17.4s
	stp	q6, q16, [x11], #32
	b	.Lmc_main
.Lmc_mid:
	cmp	x11, x13
	b.ge	.Lmc_tail
	ldr	q1, [x11]               // a
	ldr	q2, [x12], #16          // b
	trn1	v3.4s, v2.4s, v2.4s     // br
	trn2	v4.4s, v2.4s, v2.4s     // bi
	rev64	v5.4s, v1.4s
	fmul	v6.4s, v1.4s, v3.4s     // t1
	fmul	v7.4s, v5.4s, v4.4s     // t2
	eor	v7.16b, v7.16b, v26.16b
	fadd	v6.4s, v6.4s, v7.4s
	str	q6, [x11], #16
.Lmc_tail:
	cmp	x11, x10
	b.ge	.Lmc_done
	ldr	d1, [x11]
	ldr	d2, [x12], #8
	trn1	v3.2s, v2.2s, v2.2s
	trn2	v4.2s, v2.2s, v2.2s
	rev64	v5.2s, v1.2s
	fmul	v6.2s, v1.2s, v3.2s
	fmul	v7.2s, v5.2s, v4.2s
	eor	v7.8b, v7.8b, v26.8b
	fadd	v6.2s, v6.2s, v7.2s
	str	d6, [x11], #8
	b	.Lmc_tail
.Lmc_done:
	ret

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
// PackRows2: x0=z_base, x1=z_len, x2=ra_base, x3=ra_len, x4=rb_base,
// x5=rb_len, x6=step (1 or 4).
// z[i] = (float32(ra[i*step]), float32(rb[i*step])); uint8->float32 is
// exact. Stride-4 gathers use 16-byte loads, clamped to floor(len/16)
// blocks per row so they never read past the slices.
.global PackRows2
PackRows2:
	and	x7, x1, #-4             // vector element count
	cmp	x6, #4
	b.eq	.Lpk2_s4
	b	.Lpk2_s1

.Lpk2_s4:
	lsr	x8, x3, #4
	lsl	x8, x8, #2              // ra allows this many gathered elems
	cmp	x8, x7
	csel	x7, x8, x7, lt
	lsr	x8, x5, #4
	lsl	x8, x8, #2
	cmp	x8, x7
	csel	x7, x8, x7, lt
	// gather byte index (0,4,8,12) then out-of-range
	movi	v31.16b, #0xff
	mov	w9, #0x0400
	movk	w9, #0x0c08, lsl #16
	mov	v31.s[0], w9
	mov	x9, #0                  // element index
.Lpk2_s4_loop:
	cmp	x9, x7
	b.ge	.Lpk2_s4_tail
	ldr	q1, [x2], #16
	ldr	q2, [x4], #16
	tbl	v1.8b, {v1.16b}, v31.8b // bytes 0,4,8,12
	tbl	v2.8b, {v2.16b}, v31.8b
	uxtl	v1.8h, v1.8b
	uxtl	v2.8h, v2.8b
	uxtl	v1.4s, v1.4h
	uxtl	v2.4s, v2.4h
	ucvtf	v1.4s, v1.4s            // exact for bytes
	ucvtf	v2.4s, v2.4s
	zip1	v3.4s, v1.4s, v2.4s     // re0 im0 re1 im1
	zip2	v4.4s, v1.4s, v2.4s
	stp	q3, q4, [x0], #32
	add	x9, x9, #4
	b	.Lpk2_s4_loop
.Lpk2_s4_tail:
	cmp	x9, x1
	b.ge	.Lpk2_done
	ldrb	w10, [x2], #4
	ldrb	w11, [x4], #4
	ucvtf	s1, w10
	ucvtf	s2, w11
	str	s1, [x0]
	str	s2, [x0, #4]
	add	x0, x0, #8
	add	x9, x9, #1
	b	.Lpk2_s4_tail

.Lpk2_s1:
	mov	x9, #0
.Lpk2_s1_loop:
	cmp	x9, x7
	b.ge	.Lpk2_s1_tail
	ldr	s1, [x2], #4            // 4 bytes, exactly the elements used
	ldr	s2, [x4], #4
	uxtl	v1.8h, v1.8b
	uxtl	v2.8h, v2.8b
	uxtl	v1.4s, v1.4h
	uxtl	v2.4s, v2.4h
	ucvtf	v1.4s, v1.4s
	ucvtf	v2.4s, v2.4s
	zip1	v3.4s, v1.4s, v2.4s
	zip2	v4.4s, v1.4s, v2.4s
	stp	q3, q4, [x0], #32
	add	x9, x9, #4
	b	.Lpk2_s1_loop
.Lpk2_s1_tail:
	cmp	x9, x1
	b.ge	.Lpk2_done
	ldrb	w10, [x2], #1
	ldrb	w11, [x4], #1
	ucvtf	s1, w10
	ucvtf	s2, w11
	str	s1, [x0]
	str	s2, [x0, #4]
	add	x0, x0, #8
	add	x9, x9, #1
	b	.Lpk2_s1_tail
.Lpk2_done:
	ret

// ---------------------------------------------------------------------------
// PackRows1: x0=z_base, x1=z_len, x2=ra_base, x3=ra_len, x4=step.
// PackRows2 with a zero imaginary row.
.global PackRows1
PackRows1:
	and	x7, x1, #-4
	movi	v2.4s, #0               // imag lanes
	cmp	x4, #4
	b.eq	.Lpk1_s4
	b	.Lpk1_s1

.Lpk1_s4:
	lsr	x8, x3, #4
	lsl	x8, x8, #2
	cmp	x8, x7
	csel	x7, x8, x7, lt
	movi	v31.16b, #0xff
	mov	w9, #0x0400
	movk	w9, #0x0c08, lsl #16
	mov	v31.s[0], w9
	mov	x9, #0
.Lpk1_s4_loop:
	cmp	x9, x7
	b.ge	.Lpk1_s4_tail
	ldr	q1, [x2], #16
	tbl	v1.8b, {v1.16b}, v31.8b
	uxtl	v1.8h, v1.8b
	uxtl	v1.4s, v1.4h
	ucvtf	v1.4s, v1.4s
	zip1	v3.4s, v1.4s, v2.4s
	zip2	v4.4s, v1.4s, v2.4s
	stp	q3, q4, [x0], #32
	add	x9, x9, #4
	b	.Lpk1_s4_loop
.Lpk1_s4_tail:
	cmp	x9, x1
	b.ge	.Lpk1_done
	ldrb	w10, [x2], #4
	ucvtf	s1, w10
	str	s1, [x0]
	str	wzr, [x0, #4]
	add	x0, x0, #8
	add	x9, x9, #1
	b	.Lpk1_s4_tail

.Lpk1_s1:
	mov	x9, #0
.Lpk1_s1_loop:
	cmp	x9, x7
	b.ge	.Lpk1_s1_tail
	ldr	s1, [x2], #4
	uxtl	v1.8h, v1.8b
	uxtl	v1.4s, v1.4h
	ucvtf	v1.4s, v1.4s
	zip1	v3.4s, v1.4s, v2.4s
	zip2	v4.4s, v1.4s, v2.4s
	stp	q3, q4, [x0], #32
	add	x9, x9, #4
	b	.Lpk1_s1_loop
.Lpk1_s1_tail:
	cmp	x9, x1
	b.ge	.Lpk1_done
	ldrb	w10, [x2], #1
	ucvtf	s1, w10
	str	s1, [x0]
	str	wzr, [x0, #4]
	add	x0, x0, #8
	add	x9, x9, #1
	b	.Lpk1_s1_tail
.Lpk1_done:
	ret

// ---------------------------------------------------------------------------
// Untangle: x0=sa, x1=sb, x2=z, x3=n, x4=k0 (>= 1), x5=k1.
//   sa[k] = 0.5*(zk.re+zn.re, zk.im-zn.im)
//   sb[k] = 0.5*(zk.im+zn.im, zn.re-zk.re),  zn = z[n-k]
// One rounded add and one rounded halving per component, exactly the
// scalar ops (x-y == x+(-y), addition commutes bitwise).
.global Untangle
Untangle:
	mov	x9, #0x8000000000000000
	dup	v26.2d, x9              // oddsign
	fmov	v27.4s, #0.5
	sub	x6, x5, x4
	and	x13, x6, #-4
	add	x13, x13, x4            // end of 4-complex blocks
	and	x6, x6, #-2
	add	x6, x6, x4              // end of 2-complex blocks
	add	x7, x0, x4, lsl #3      // sa ptr
	add	x8, x1, x4, lsl #3      // sb ptr
	add	x10, x2, x4, lsl #3     // zk ptr
	sub	x11, x3, x4             // n-k0
	sub	x11, x11, #1            // n-k0-1
	add	x11, x2, x11, lsl #3    // zn load ptr (descending pairs)
	mov	x12, x4                 // k
.Lut_loop4:
	cmp	x12, x13
	b.ge	.Lut_loop
	ldp	q1, q10, [x10], #32     // zk pairs
	ldp	q12, q2, [x11, #-16]    // zn pairs, descending
	sub	x11, x11, #32
	ext	v2.16b, v2.16b, v2.16b, #8 // zn for k, k+1
	ext	v12.16b, v12.16b, v12.16b, #8 // zn for k+2, k+3
	eor	v3.16b, v2.16b, v26.16b
	fadd	v3.4s, v1.4s, v3.4s
	fmul	v3.4s, v3.4s, v27.4s
	eor	v13.16b, v12.16b, v26.16b
	fadd	v13.4s, v10.4s, v13.4s
	fmul	v13.4s, v13.4s, v27.4s
	stp	q3, q13, [x7], #32      // sa
	rev64	v4.4s, v1.4s
	eor	v4.16b, v4.16b, v26.16b
	rev64	v5.4s, v2.4s
	fadd	v4.4s, v4.4s, v5.4s
	fmul	v4.4s, v4.4s, v27.4s
	rev64	v14.4s, v10.4s
	eor	v14.16b, v14.16b, v26.16b
	rev64	v15.4s, v12.4s
	fadd	v14.4s, v14.4s, v15.4s
	fmul	v14.4s, v14.4s, v27.4s
	stp	q4, q14, [x8], #32      // sb
	add	x12, x12, #4
	b	.Lut_loop4
.Lut_loop:
	cmp	x12, x6
	b.ge	.Lut_tail
	ldr	q1, [x10], #16          // zk, zk+1
	ldr	q2, [x11], #-16         // z[n-k-1], z[n-k]
	ext	v2.16b, v2.16b, v2.16b, #8 // (z[n-k], z[n-k-1])
	eor	v3.16b, v2.16b, v26.16b // (zn.re, -zn.im)
	fadd	v3.4s, v1.4s, v3.4s
	fmul	v3.4s, v3.4s, v27.4s
	str	q3, [x7], #16           // sa
	rev64	v4.4s, v1.4s
	eor	v4.16b, v4.16b, v26.16b // (zk.im, -zk.re)
	rev64	v5.4s, v2.4s            // (zn.im, zn.re)
	fadd	v4.4s, v4.4s, v5.4s
	fmul	v4.4s, v4.4s, v27.4s
	str	q4, [x8], #16           // sb
	add	x12, x12, #2
	b	.Lut_loop
.Lut_tail:
	cmp	x12, x5
	b.ge	.Lut_done
	ldr	d1, [x10], #8           // zk
	sub	x13, x3, x12
	ldr	d2, [x2, x13, lsl #3]   // zn
	eor	v3.8b, v2.8b, v26.8b
	fadd	v3.2s, v1.2s, v3.2s
	fmul	v3.2s, v3.2s, v27.2s
	str	d3, [x7], #8
	rev64	v4.2s, v1.2s
	eor	v4.8b, v4.8b, v26.8b
	rev64	v5.2s, v2.2s
	fadd	v4.2s, v4.2s, v5.2s
	fmul	v4.2s, v4.2s, v27.2s
	str	d4, [x8], #8
	add	x12, x12, #1
	b	.Lut_tail
.Lut_done:
	ret

// ---------------------------------------------------------------------------
// CombineLow: x0=z_base, x1=len, x2=sa, x3=sb.
// z[k] = (sa.re - sb.im, sa.im + sb.re) = sa + (rev64(sb) ^ evensign).
.global CombineLow
CombineLow:
	mov	w9, #0x80000000
	dup	v26.2d, x9              // evensign
	and	x7, x1, #-4
	add	x8, x0, x7, lsl #3      // end of 4-complex blocks
	and	x4, x1, #-2
	add	x5, x0, x4, lsl #3
	add	x6, x0, x1, lsl #3
.Lcl_main4:
	cmp	x0, x8
	b.ge	.Lcl_main
	ldp	q1, q10, [x2], #32
	ldp	q2, q11, [x3], #32
	rev64	v2.4s, v2.4s
	eor	v2.16b, v2.16b, v26.16b
	fadd	v1.4s, v1.4s, v2.4s
	rev64	v11.4s, v11.4s
	eor	v11.16b, v11.16b, v26.16b
	fadd	v10.4s, v10.4s, v11.4s
	stp	q1, q10, [x0], #32
	b	.Lcl_main4
.Lcl_main:
	cmp	x0, x5
	b.ge	.Lcl_tail
	ldr	q1, [x2], #16
	ldr	q2, [x3], #16
	rev64	v2.4s, v2.4s
	eor	v2.16b, v2.16b, v26.16b
	fadd	v1.4s, v1.4s, v2.4s
	str	q1, [x0], #16
.Lcl_tail:
	cmp	x0, x6
	b.ge	.Lcl_done
	ldr	d1, [x2], #8
	ldr	d2, [x3], #8
	rev64	v2.2s, v2.2s
	eor	v2.8b, v2.8b, v26.8b
	fadd	v1.2s, v1.2s, v2.2s
	str	d1, [x0], #8
	b	.Lcl_tail
.Lcl_done:
	ret

// ---------------------------------------------------------------------------
// CombineHigh: x0=z, x1=sa, x2=sb, x3=n, x4=hw.
// z[k] = (sa[m].re + sb[m].im, sb[m].re - sa[m].im), m = n-k, k in [hw, n)
//      = rev64(sb[m]) + (sa[m] ^ oddsign).
.global CombineHigh
CombineHigh:
	mov	x9, #0x8000000000000000
	dup	v26.2d, x9              // oddsign
	sub	x5, x3, x4              // count = n-hw
	and	x13, x5, #-4
	add	x14, x0, x4, lsl #3
	add	x14, x14, x13, lsl #3   // z end of 4-complex blocks
	and	x6, x5, #-2
	add	x7, x0, x4, lsl #3      // z ptr at k=hw
	add	x8, x7, x6, lsl #3      // z end of 2-complex blocks
	add	x9, x0, x3, lsl #3      // z true end
	sub	x10, x3, x4             // m = n-hw
	sub	x10, x10, #1            // m-1
	add	x11, x1, x10, lsl #3    // sa load ptr (descending pairs)
	add	x12, x2, x10, lsl #3    // sb load ptr
.Lch_main4:
	cmp	x7, x14
	b.ge	.Lch_main
	ldp	q10, q1, [x11, #-16]    // sa[m-3..m-2], sa[m-1..m]
	sub	x11, x11, #32
	ldp	q11, q2, [x12, #-16]
	sub	x12, x12, #32
	ext	v1.16b, v1.16b, v1.16b, #8 // sa[m], sa[m-1] for k, k+1
	ext	v2.16b, v2.16b, v2.16b, #8
	rev64	v2.4s, v2.4s
	eor	v1.16b, v1.16b, v26.16b
	fadd	v1.4s, v2.4s, v1.4s
	ext	v10.16b, v10.16b, v10.16b, #8 // sa[m-2], sa[m-3] for k+2, k+3
	ext	v11.16b, v11.16b, v11.16b, #8
	rev64	v11.4s, v11.4s
	eor	v10.16b, v10.16b, v26.16b
	fadd	v10.4s, v11.4s, v10.4s
	stp	q1, q10, [x7], #32
	b	.Lch_main4
.Lch_main:
	cmp	x7, x8
	b.ge	.Lch_tail
	ldr	q1, [x11], #-16         // sa[m-1], sa[m]
	ldr	q2, [x12], #-16         // sb[m-1], sb[m]
	ext	v1.16b, v1.16b, v1.16b, #8 // sa[m], sa[m-1] for k, k+1
	ext	v2.16b, v2.16b, v2.16b, #8
	rev64	v2.4s, v2.4s            // (sb.im, sb.re)
	eor	v1.16b, v1.16b, v26.16b // (sa.re, -sa.im)
	fadd	v1.4s, v2.4s, v1.4s
	str	q1, [x7], #16
	b	.Lch_main
.Lch_tail:
	cmp	x7, x9
	b.ge	.Lch_done
	sub	x10, x9, x7             // bytes left = (n-k)*8
	lsr	x10, x10, #3            // m = n-k
	ldr	d1, [x1, x10, lsl #3]
	ldr	d2, [x2, x10, lsl #3]
	rev64	v2.2s, v2.2s
	eor	v1.8b, v1.8b, v26.8b
	fadd	v1.2s, v2.2s, v1.2s
	str	d1, [x7], #8
	b	.Lch_tail
.Lch_done:
	ret

// ---------------------------------------------------------------------------
// EmitRe: x0=dst, x1=len, x2=z, x3=add flag.
// dst = (or +=) real parts of z, one exact f32 add per element when
// accumulating (addition commutes bitwise).
.global EmitRe
EmitRe:
	and	x7, x1, #-8
	add	x8, x0, x7, lsl #2      // dst end of 8-element blocks
	and	x4, x1, #-4
	add	x5, x0, x4, lsl #2      // dst end of 4-element blocks
	add	x6, x0, x1, lsl #2      // dst true end
.Ler_main8:
	cmp	x0, x8
	b.ge	.Ler_main
	ldp	q1, q2, [x2], #32
	ldp	q10, q11, [x2], #32
	uzp1	v3.4s, v1.4s, v2.4s     // re lanes
	uzp1	v12.4s, v10.4s, v11.4s
	cbz	x3, .Ler_store8
	ldp	q0, q9, [x0]
	fadd	v3.4s, v0.4s, v3.4s
	fadd	v12.4s, v9.4s, v12.4s
.Ler_store8:
	stp	q3, q12, [x0], #32
	b	.Ler_main8
.Ler_main:
	cmp	x0, x5
	b.ge	.Ler_tail
	ldp	q1, q2, [x2], #32
	uzp1	v3.4s, v1.4s, v2.4s     // re lanes
	cbz	x3, .Ler_store
	ldr	q0, [x0]
	fadd	v3.4s, v0.4s, v3.4s
.Ler_store:
	str	q3, [x0], #16
.Ler_tail:
	cmp	x0, x6
	b.ge	.Ler_done
	ldr	s1, [x2], #8
	cbz	x3, .Ler_tstore
	ldr	s0, [x0]
	fadd	s1, s0, s1
.Ler_tstore:
	str	s1, [x0], #4
	b	.Ler_tail
.Ler_done:
	ret

// ---------------------------------------------------------------------------
// EmitIm: x0=dst, x1=len, x2=z, x3=add flag. Imaginary-part variant.
.global EmitIm
EmitIm:
	and	x7, x1, #-8
	add	x8, x0, x7, lsl #2
	and	x4, x1, #-4
	add	x5, x0, x4, lsl #2
	add	x6, x0, x1, lsl #2
.Lei_main8:
	cmp	x0, x8
	b.ge	.Lei_main
	ldp	q1, q2, [x2], #32
	ldp	q10, q11, [x2], #32
	uzp2	v3.4s, v1.4s, v2.4s     // im lanes
	uzp2	v12.4s, v10.4s, v11.4s
	cbz	x3, .Lei_store8
	ldp	q0, q9, [x0]
	fadd	v3.4s, v0.4s, v3.4s
	fadd	v12.4s, v9.4s, v12.4s
.Lei_store8:
	stp	q3, q12, [x0], #32
	b	.Lei_main8
.Lei_main:
	cmp	x0, x5
	b.ge	.Lei_tail
	ldp	q1, q2, [x2], #32
	uzp2	v3.4s, v1.4s, v2.4s     // im lanes
	cbz	x3, .Lei_store
	ldr	q0, [x0]
	fadd	v3.4s, v0.4s, v3.4s
.Lei_store:
	str	q3, [x0], #16
.Lei_tail:
	cmp	x0, x6
	b.ge	.Lei_done
	ldr	s1, [x2, #4]
	add	x2, x2, #8
	cbz	x3, .Lei_tstore
	ldr	s0, [x0]
	fadd	s1, s0, s1
.Lei_tstore:
	str	s1, [x0], #4
	b	.Lei_tail
.Lei_done:
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
