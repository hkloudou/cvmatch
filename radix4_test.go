package cvmatch

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// naiveDFT is the float64 O(n^2) oracle.
func naiveDFT(x []complex64, inverse bool) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	sign := -2 * math.Pi
	if inverse {
		sign = 2 * math.Pi
	}
	for k := 0; k < n; k++ {
		var acc complex128
		for j := 0; j < n; j++ {
			ang := sign * float64(k*j%n) / float64(n)
			acc += complex128(x[j]) * cmplx.Exp(complex(0, ang))
		}
		out[k] = acc
	}
	return out
}

func randSlab(n int, seed int64) []complex64 {
	rng := rand.New(rand.NewSource(seed))
	a := make([]complex64, n)
	for i := range a {
		a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}
	return a
}

func TestFFTR4MatchesOracle(t *testing.T) {
	for n := 1; n <= 4096; n *= 2 {
		tab := makeR4Tab(n)
		for _, inverse := range []bool{false, true} {
			x := randSlab(n, int64(n)+7)
			got := append([]complex64(nil), x...)
			fftR4(got, tab, inverse)
			want := naiveDFT(x, inverse)
			// f32 FFT error grows ~sqrt(log n); scale by the L2 norm.
			var norm float64
			for _, w := range want {
				norm += real(w)*real(w) + imag(w)*imag(w)
			}
			tol := 3e-7 * math.Sqrt(norm) * math.Sqrt(math.Log2(float64(n))+1) * 4
			for k := range got {
				d := cmplx.Abs(complex128(got[k]) - want[k])
				if d > tol {
					t.Fatalf("n=%d inverse=%v k=%d: |Δ|=%.3e > tol %.3e", n, inverse, k, d, tol)
				}
			}
		}
	}
}

// The radix-4 engine and the production radix-2 engine approximate the
// same DFT; their mutual deviation must stay in the same error class
// (this also pins the B/C quarter orientation crisply — a swap error
// yields O(1) deviations, not O(eps)).
func TestFFTR4NearRadix2(t *testing.T) {
	for n := 8; n <= 4096; n *= 2 {
		tab := makeR4Tab(n)
		ft := fftTables(n)
		for _, inverse := range []bool{false, true} {
			x := randSlab(n, int64(n)+31)
			r4 := append([]complex64(nil), x...)
			r2 := append([]complex64(nil), x...)
			fftR4(r4, tab, inverse)
			fftGo(r2, ft.tw, ft.pairs, inverse)
			var norm float64
			for _, w := range r2 {
				norm += float64(real(w))*float64(real(w)) + float64(imag(w))*float64(imag(w))
			}
			tol := 1e-6 * math.Sqrt(norm)
			for k := range r4 {
				d := cmplx.Abs(complex128(r4[k]) - complex128(r2[k]))
				if d > tol {
					t.Fatalf("n=%d inverse=%v k=%d: radix4 vs radix2 |Δ|=%.3e > %.3e",
						n, inverse, k, d, tol)
				}
			}
		}
	}
}

func TestFFTR4RoundTrip(t *testing.T) {
	for _, n := range []int{2, 4, 8, 64, 512, 1024} {
		tab := makeR4Tab(n)
		x := randSlab(n, int64(n)+3)
		a := append([]complex64(nil), x...)
		fftR4(a, tab, false)
		fftR4(a, tab, true)
		inv := 1 / float32(n)
		for i := range a {
			a[i] = complex(real(a[i])*inv, imag(a[i])*inv)
		}
		for i := range a {
			if d := cmplx.Abs(complex128(a[i]) - complex128(x[i])); d > 2e-5 {
				t.Fatalf("n=%d i=%d roundtrip |Δ|=%.3e", n, i, d)
			}
		}
	}
}

// Early 1-D cost probes for the 7.2 ship decision (purego leg): the
// pipeline A/B is the real gate, these bound the per-butterfly cost.
func Benchmark1DRadix2(b *testing.B) {
	for _, n := range []int{512, 1024} {
		ft := fftTables(n)
		a := randSlab(n, 1)
		b.Run(itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fftGo(a, ft.tw, ft.pairs, i&1 == 1)
			}
		})
	}
}

func Benchmark1DRadix4(b *testing.B) {
	for _, n := range []int{512, 1024} {
		tab := makeR4Tab(n)
		a := randSlab(n, 1)
		b.Run(itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fftR4(a, tab, i&1 == 1)
			}
		})
	}
}

func itoa(n int) string {
	if n == 512 {
		return "512"
	}
	return "1024"
}
