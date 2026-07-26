package cvmatch

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// fma32Exact computes float32(a*b+c) with one rounding via exact
// arbitrary-precision arithmetic — the oracle for fma32.
func fma32Exact(a, b, c float32) float32 {
	const prec = 200
	x := new(big.Float).SetPrec(prec).SetFloat64(float64(a))
	y := new(big.Float).SetPrec(prec).SetFloat64(float64(b))
	z := new(big.Float).SetPrec(prec).SetFloat64(float64(c))
	x.Mul(x, y).Add(x, z)
	f, _ := x.Float32() // ToNearestEven
	return f
}

func TestFMA32DirectedCases(t *testing.T) {
	cases := [][3]float32{
		// The double-rounding counterexample from the determinism
		// framework: float32(math.FMA(f64...)) gets this one wrong.
		{math.Float32frombits(0x3F800003), 1.5, float32(math.Ldexp(1, -60))},
		{1, 1, 0}, {1, 1, -1}, {0, 0, 0},
		{-0, 0, -0}, {1, -1, 1},
		// exact cancellation: a*b == -c
		{3, 5, -15},
		// c dominates / p dominates
		{1e-30, 1e-30, 1}, {1e15, 1e15, -1},
		// subnormal territory
		{math.Float32frombits(1), math.Float32frombits(1), 0},
		{math.SmallestNonzeroFloat32, 0.5, 0},
		{math.SmallestNonzeroFloat32, 0.5, math.SmallestNonzeroFloat32},
		// f32 overflow via the product
		{math.MaxFloat32, 2, 0}, {math.MaxFloat32, -2, 0},
		{math.MaxFloat32, 1, math.MaxFloat32},
	}
	for _, tc := range cases {
		got := fma32(tc[0], tc[1], tc[2])
		want := fma32Exact(tc[0], tc[1], tc[2])
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Errorf("fma32(%x, %x, %x) = %x (%g), want %x (%g)",
				tc[0], tc[1], tc[2], got, got, want, want)
		}
	}
	// And prove the counterexample really is a counterexample for the
	// naive double-rounded version, so this test would catch a future
	// "simplification" back to it.
	a, b, c := math.Float32frombits(0x3F800003), float32(1.5), float32(math.Ldexp(1, -60))
	naive := float32(math.FMA(float64(a), float64(b), float64(c)))
	if math.Float32bits(naive) == math.Float32bits(fma32Exact(a, b, c)) {
		t.Fatalf("counterexample no longer distinguishes naive double rounding — replace it")
	}
}

func TestFMA32RandomCrossCheck(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	f := func() float32 {
		// full-range bit patterns, skipping NaN/Inf (kernel domain is
		// finite); includes subnormals and both signs
		for {
			v := math.Float32frombits(rng.Uint32())
			if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
				return v
			}
		}
	}
	for i := 0; i < 2_000_000; i++ {
		a, b, c := f(), f(), f()
		got := fma32(a, b, c)
		want := fma32Exact(a, b, c)
		if math.Float32bits(got) != math.Float32bits(want) &&
			!(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
			t.Fatalf("i=%d: fma32(%x, %x, %x) = %x, want %x",
				i, a, b, c, got, want)
		}
	}
}
