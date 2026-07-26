package cvmatch

import (
	"math"
	"math/rand"
	"testing"

	"github.com/hkloudou/cvmatch/internal/simd"
)

// TestStatsCapBounds verifies the per-channel exact-statistics caps both
// ways: the recorded cap for each cn is the largest area with
// cn·65025·area² < 2^63 (so cn=1 admits large grayscale templates the
// cn=4 worst case would reject — a 3000x2000 gray template is legal),
// and cap+1 overflows.
func TestStatsCapBounds(t *testing.T) {
	const limit int64 = 1<<63 - 1
	for cn := 1; cn <= 4; cn++ {
		a := statsCap(cn)
		if v := int64(cn) * 65025; a*a > limit/v {
			t.Errorf("cn=%d: cap %d overflows cn·65025·area²", cn, a)
		} else if b := a + 1; b*b <= limit/v {
			t.Errorf("cn=%d: cap %d is not tight (cap+1 still fits)", cn, b)
		}
	}
	if a := int64(3000) * 2000; a > statsCap(1) {
		t.Errorf("3000x2000 single-channel template must be inside the cn=1 cap")
	}
	defer func() {
		if recover() == nil {
			t.Errorf("cn=4 template above the cap must panic")
		}
	}()
	matchU8(nil, 2500*4, 2500, 2500, nil, 2500*4, 2500, 2500, 4, 4, 1, nil)
}

// SpillStats4 must reproduce the scalar cn=3/4 spill bit-for-bit. Data
// is synthesized as uniform columns (all th pixels of column x share
// one byte per channel), which keeps every pipeline invariant — column
// sums <= 255·th, window sums <= 255·area, and idiff >= 0 by
// Cauchy-Schwarz — while reaching the exact stats-cap extremes the
// conversions must survive.
func TestSpillStats4MatchesScalar(t *testing.T) {
	if !simd.Enabled {
		t.Skip("kernels disabled")
	}
	rng := rand.New(rand.NewSource(7))
	const stride = 64
	for _, cn := range []int{3, 4} {
		thMax := 11008
		if cn == 4 {
			thMax = 8256
		}
		for _, tc := range []struct{ tw, th int }{
			{3, 5}, {96, 32}, {2048, 2048}, {181, thMax},
		} {
			area := int64(tc.tw) * int64(tc.th)
			if area > statsCap(cn) {
				continue
			}
			four := cn == 4
			n := 61 // non-multiple-of-4 so the Go tail also runs
			cols := n + tc.tw
			lo := make([]int32, cols*4)
			lo2 := make([]int64, cols)
			for x := 0; x < cols; x++ {
				var sq int64
				for k := 0; k < 4; k++ {
					v := rng.Int63n(256)
					lo[x*4+k] = int32(v * int64(tc.th))
					if k < cn {
						sq += v * v
					}
				}
				lo2[x] = sq * int64(tc.th)
			}
			hi := lo[tc.tw*4:]
			hi2 := lo2[tc.tw:]
			var tsum [4]int64
			for k := 0; k < cn; k++ {
				tsum[k] = rng.Int63n(255*area + 1)
			}
			var sA [4]int64
			var s2A int64
			for x := 0; x < tc.tw; x++ {
				for k := 0; k < cn; k++ {
					sA[k] += int64(lo[x*4+k])
				}
				s2A += lo2[x]
			}
			sB, s2B := sA, s2A
			wantWt := make([]float32, 3*stride)
			gotWt := make([]float32, 3*stride)
			vns := n &^ 3

			// scalar reference: the exact case-3/4 loop shapes
			{
				s0, s1, c2, s3, t2 := sA[0], sA[1], sA[2], sA[3], s2A
				for i := 0; i < vns; i++ {
					cross := s0*tsum[0] + s1*tsum[1] + c2*tsum[2]
					sq := s0*s0 + s1*s1 + c2*c2
					if four {
						cross += s3 * tsum[3]
						sq += s3 * s3
					}
					wantWt[i] = float32(float64(cross))
					wantWt[stride+i] = float32(float64(area*t2 - sq))
					wantWt[2*stride+i] = float32(float64(t2))
					j := i * 4
					s0 += int64(hi[j] - lo[j])
					s1 += int64(hi[j+1] - lo[j+1])
					c2 += int64(hi[j+2] - lo[j+2])
					if four {
						s3 += int64(hi[j+3] - lo[j+3])
					}
					t2 += hi2[i] - lo2[i]
				}
				sA[0], sA[1], sA[2], s2A = s0, s1, c2, t2
				if four {
					sA[3] = s3
				}
			}
			sK := sB
			if !four {
				sK[3] = 0
				sA[3] = 0
			}
			ns2 := simd.SpillStats4(gotWt[:vns], stride, lo, hi, lo2, hi2,
				&sK, &tsum, s2B, area, four)
			if sK != sA || ns2 != s2A {
				t.Fatalf("cn=%d %dx%d: advanced sums kernel %v/%d want %v/%d",
					cn, tc.tw, tc.th, sK, ns2, sA, s2A)
			}
			for i := 0; i < 3*stride; i++ {
				if math.Float32bits(gotWt[i]) != math.Float32bits(wantWt[i]) {
					t.Fatalf("cn=%d %dx%d lane elem %d: kernel %x want %x",
						cn, tc.tw, tc.th, i, gotWt[i], wantWt[i])
				}
			}
		}
	}
}
