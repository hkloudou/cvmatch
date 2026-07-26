package cvmatch

import "math"

// fma32 returns a*b+c rounded ONCE to float32 — the scalar twin of the
// asm kernels' single-precision VFMADDPS. A bare
// float32(math.FMA(float64(a), float64(b), float64(c))) double-rounds:
// the f64 FMA rounds the exact result to 53 bits and the float32
// conversion rounds again, which differs from one 24-bit rounding when
// the first rounding lands exactly on a 24-bit halfway point
// (counterexample pinned in the tests: a=0x1.000006p0, b=1.5, c=2^-60).
//
// The repair is round-to-odd (Boldo–Melquiond): the product of two
// float32s is EXACT in float64 (24+24 <= 53 bits), so the only rounding
// happens in the f64 addition p+c. TwoSum recovers that addition's exact
// residual e; when e != 0 and the rounded sum s has an even low mantissa
// bit, s is nudged one ulp toward e, making the low bit a sticky bit.
// Round-to-odd at 53 bits followed by one rounding to 24 bits equals the
// correctly rounded 24-bit result (valid whenever the wide precision has
// at least 2·24+2 bits; 53 >= 50). Pure sequenced f64 ops — bit-identical
// on every platform, no FMA hardware required.
func fma32(a, b, c float32) float32 {
	p := float64(a) * float64(b) // exact
	c64 := float64(c)
	s := p + c64
	// TwoSum (Knuth, branchless): e = (p + c64) - s exactly.
	t := s - p
	e := (p - (s - t)) + (c64 - t)
	if e != 0 && !math.IsInf(s, 0) {
		bits := math.Float64bits(s)
		if bits&1 == 0 {
			// s == 0 implies e == 0 (a two-float sum rounding to zero
			// is exact), so s's sign is meaningful here.
			if (e > 0) == (s > 0) {
				bits++ // odd neighbor away from zero
			} else {
				bits-- // odd neighbor toward zero
			}
			s = math.Float64frombits(bits)
		}
	}
	return float32(s)
}
