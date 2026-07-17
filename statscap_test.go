package cvmatch

import "testing"

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
