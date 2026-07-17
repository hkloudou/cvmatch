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

// FFT kernels: the radix-2 stage cascade, column-direction butterflies
// (single stage and fused stage pairs), and the conjugate spectrum
// multiply.

.text

// ---------------------------------------------------------------------------
// FFTStages: x0=a_base, x1=n (power of two >= 8), x2=tw_base, x3=inverse(0/1)
// Runs the full radix-2 cascade over bit-reversed data. Stages half=1,2 are
// fused into one pass (every 4-complex group is closed under both stages),
// then the generic cascade runs from half=4 in 2-complex steps.
// Register map (FFTStages):
//   x0=a_base  x1=n  x2=tw_base  x3=inverse  x9=cursor  x10=count
//   x11..x14=stage/block/j counters (cascade)
//   v29=s (sign lanes when inverse)  v26=evensign  v25=highsign
//   v27/v28=wr1/wi1 (tw[1] broadcast)  v23/v24=w2 re/im pairs (tw[2],tw[3])
//   loop scratch: v0..v7 data quads, v16..v22 butterfly temporaries
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
