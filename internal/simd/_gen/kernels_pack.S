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

// Data-movement kernels: byte->complex packing, spectrum
// untangle/combine for the 2-for-1 real FFT, and the inverse-transform
// result emit.

.text

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
