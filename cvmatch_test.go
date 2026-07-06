package cvmatch

import (
	"image"
	"math"
	"math/rand"
	"testing"
)

// refMatch is a brute-force float64 TM_CCOEFF_NORMED reference implementing
// OpenCV's formulas (including the rounding guard and 1.125 saturation band)
// directly from the definition, in O(rw*rh*tw*th*cn).
func refMatch(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step int) []float64 {
	area := float64(tw * th)
	invArea := 1 / area
	mean := make([]float64, cn)
	templNorm := 0.0
	for k := 0; k < cn; k++ {
		s, s2 := 0.0, 0.0
		for y := 0; y < th; y++ {
			for x := 0; x < tw; x++ {
				v := float64(tpl[y*tstride+x*step+k])
				s += v
				s2 += v * v
			}
		}
		mean[k] = s * invArea
		templNorm += s2*invArea - mean[k]*mean[k]
	}
	rw, rh := iw-tw+1, ih-th+1
	res := make([]float64, rw*rh)
	if templNorm < 2.220446049250313e-16 {
		for i := range res {
			res[i] = 1
		}
		return res
	}
	templNorm = math.Sqrt(templNorm * area)
	const fltEps = 1.1920929e-07
	for y := 0; y < rh; y++ {
		for x := 0; x < rw; x++ {
			// corr accumulated in float32 like the FFT path / OpenCV DFT.
			var corr float64
			wndSum := make([]float64, cn)
			wndSum2 := 0.0
			for dy := 0; dy < th; dy++ {
				for dx := 0; dx < tw; dx++ {
					for k := 0; k < cn; k++ {
						iv := float64(img[(y+dy)*istride+(x+dx)*step+k])
						tv := float64(tpl[dy*tstride+dx*step+k])
						corr += iv * tv
						wndSum[k] += iv
						wndSum2 += iv * iv
					}
				}
			}
			num := corr
			wndMean2 := 0.0
			for k := 0; k < cn; k++ {
				wndMean2 += wndSum[k] * wndSum[k]
				num -= wndSum[k] * mean[k]
			}
			wndMean2 *= invArea
			diff2 := math.Max(wndSum2-wndMean2, 0)
			var den float64
			if diff2 > math.Min(0.5, 10*fltEps*wndSum2) {
				den = math.Sqrt(diff2) * templNorm
			}
			switch {
			case math.Abs(num) < den:
				num /= den
			case math.Abs(num) < den*1.125:
				num = math.Copysign(1, num)
			default:
				num = 0
			}
			res[y*rw+x] = num
		}
	}
	return res
}

func randPix(n int, rng *rand.Rand) []uint8 {
	p := make([]uint8, n)
	for i := range p {
		p[i] = uint8(rng.Intn(256))
	}
	return p
}

func TestAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cases := []struct{ iw, ih, tw, th, cn, ipad, tpad int }{
		{97, 61, 17, 13, 1, 0, 0},
		{97, 61, 17, 13, 4, 0, 0},
		{64, 64, 64, 64, 1, 0, 0},   // template == image
		{50, 40, 1, 1, 4, 0, 0},     // 1x1 template
		{300, 200, 31, 29, 1, 7, 3}, // strided rows (sub-image case)
		{40, 300, 33, 257, 1, 0, 0}, // tall template, tiny rw
		{128, 96, 5, 4, 3, 0, 0},    // 3-channel
	}
	for ci, c := range cases {
		istride, tstride := (c.iw+c.ipad)*c.cn, (c.tw+c.tpad)*c.cn
		img := randPix(istride*c.ih, rng)
		tpl := randPix(tstride*c.th, rng)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1
		got := make([]float32, rw*rh)
		minV, minX, minY, maxV, maxX, maxY := matchU8(img, istride, c.iw, c.ih, tpl, tstride, c.tw, c.th, c.cn, c.cn, 4, got)
		want := refMatch(img, istride, c.iw, c.ih, tpl, tstride, c.tw, c.th, c.cn, c.cn)

		worst := 0.0
		for i := range want {
			d := math.Abs(float64(got[i]) - want[i])
			if d > worst {
				worst = d
			}
		}
		if worst > 1e-4 {
			t.Fatalf("case %d: max abs error %g vs reference", ci, worst)
		}
		// The scan must agree with a reference row-major first-occurrence scan
		// over our own values (locations of near-ties may legally differ from
		// the float64 reference, so verify against got itself).
		rminV, rmaxV := float32(math.Inf(1)), float32(math.Inf(-1))
		rminI, rmaxI := 0, 0
		for i, v := range got {
			if v < rminV {
				rminV, rminI = v, i
			}
			if v > rmaxV {
				rmaxV, rmaxI = v, i
			}
		}
		if minV != rminV || maxV != rmaxV || minY*rw+minX != rminI || maxY*rw+maxX != rmaxI {
			t.Fatalf("case %d: min/max scan mismatch: got (%v @%d,%d)/(%v @%d,%d) want idx %d/%d",
				ci, minV, minX, minY, maxV, maxX, maxY, rminI, rmaxI)
		}
	}
}

func makeParentSub(w, h, sw, sh, sx, sy int, rng *rand.Rand) (*image.RGBA, *image.RGBA) {
	parent := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(parent.Pix); i += 4 {
		parent.Pix[i] = uint8(rng.Intn(256))
		parent.Pix[i+1] = uint8(rng.Intn(256))
		parent.Pix[i+2] = uint8(rng.Intn(256))
		parent.Pix[i+3] = 255
	}
	sub := image.NewRGBA(image.Rect(0, 0, sw, sh))
	for y := 0; y < sh; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+sw*4],
			parent.Pix[(sy+y)*parent.Stride+sx*4:])
	}
	return parent, sub
}

func TestPlantedTemplate(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	parent, sub := makeParentSub(640, 400, 64, 48, 123, 77, rng)

	for name, fn := range map[string]func(image.Image, image.Image) (float32, int, int, float32, int, int){
		"Match": Match, "MatchGray": MatchGray,
	} {
		_, _, _, maxV, maxX, maxY := fn(parent, sub)
		if maxX != 123 || maxY != 77 || maxV < 0.999 {
			t.Fatalf("%s: expected match at (123,77) with maxVal~1, got (%d,%d) %v", name, maxX, maxY, maxV)
		}
	}
}

func TestSubImageInput(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	parent, _ := makeParentSub(320, 240, 1, 1, 0, 0, rng)
	region := parent.SubImage(image.Rect(50, 60, 130, 140)).(*image.RGBA)
	_, _, _, maxV, maxX, maxY := Match(parent, region)
	if maxX != 50 || maxY != 60 || maxV < 0.999 {
		t.Fatalf("sub-image region should match at (50,60): got (%d,%d) %v", maxX, maxY, maxV)
	}
}

func TestYCbCrGray(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	yc := image.NewYCbCr(image.Rect(0, 0, 200, 150), image.YCbCrSubsampleRatio420)
	for i := range yc.Y {
		yc.Y[i] = uint8(rng.Intn(256))
	}
	sub := image.NewGray(image.Rect(0, 0, 40, 30))
	for y := 0; y < 30; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+40], yc.Y[(20+y)*yc.YStride+60:])
	}
	_, _, _, maxV, maxX, maxY := MatchGray(yc, sub)
	if maxX != 60 || maxY != 20 || maxV < 0.999 {
		t.Fatalf("expected match at (60,20), got (%d,%d) %v", maxX, maxY, maxV)
	}
}

func TestFlatTemplate(t *testing.T) {
	parent := image.NewRGBA(image.Rect(0, 0, 100, 100))
	sub := image.NewRGBA(image.Rect(0, 0, 10, 10)) // all zeros: zero variance
	minV, minX, minY, maxV, maxX, maxY := Match(parent, sub)
	if minV != 1 || maxV != 1 || minX != 0 || minY != 0 || maxX != 0 || maxY != 0 {
		t.Fatalf("flat template should yield all-ones result, got %v/%v", minV, maxV)
	}
}

func TestPanicsOnBadInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic when sub is larger than parent")
		}
	}()
	Match(image.NewRGBA(image.Rect(0, 0, 10, 10)), image.NewRGBA(image.Rect(0, 0, 20, 20)))
}

// TestConstantAlphaSkip verifies the claim behind Match's cn=3 fast path:
// with a per-image constant alpha plane, processing 3 of 4 channels yields
// the same response map as the full 4-channel computation.
func TestConstantAlphaSkip(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	iw, ih, tw, th := 120, 90, 24, 18
	img, tpl := randPix(iw*ih*4, rng), randPix(tw*th*4, rng)
	rw, rh := iw-tw+1, ih-th+1
	// different constants per image, including the 0/255 extremes
	for _, pair := range [][2]uint8{{200, 55}, {0, 255}, {255, 0}, {128, 128}} {
		for i := 3; i < len(img); i += 4 {
			img[i] = pair[0]
		}
		for i := 3; i < len(tpl); i += 4 {
			tpl[i] = pair[1]
		}
		got3 := make([]float32, rw*rh)
		got4 := make([]float32, rw*rh)
		matchU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 3, 4, 4, got3)
		matchU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 4, 4, 4, got4)
		for i := range got3 {
			if d := math.Abs(float64(got3[i]) - float64(got4[i])); d > 1e-5 {
				t.Fatalf("alpha=%v: cn=3 vs cn=4 mismatch at %d: %v vs %v", pair, i, got3[i], got4[i])
			}
		}
	}
}

// TestVaryingAlpha makes sure Match falls back to the full 4-channel path
// and still agrees with the float64 reference when alpha carries signal.
func TestVaryingAlpha(t *testing.T) {
	rng := rand.New(rand.NewSource(23))
	parent := image.NewRGBA(image.Rect(0, 0, 100, 80))
	copy(parent.Pix, randPix(len(parent.Pix), rng))
	sub := image.NewRGBA(image.Rect(0, 0, 20, 16))
	for y := 0; y < 16; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+20*4], parent.Pix[(30+y)*parent.Stride+40*4:])
	}
	_, _, _, maxV, maxX, maxY := Match(parent, sub)
	if maxX != 40 || maxY != 30 || maxV < 0.999 {
		t.Fatalf("expected match at (40,30), got (%d,%d) %v", maxX, maxY, maxV)
	}
	want := refMatch(parent.Pix, parent.Stride, 100, 80, sub.Pix, sub.Stride, 20, 16, 4, 4)
	got := make([]float32, len(want))
	matchU8(parent.Pix, parent.Stride, 100, 80, sub.Pix, sub.Stride, 20, 16, 4, 4, 4, got)
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); d > 1e-4 {
			t.Fatalf("varying-alpha mismatch at %d: %v vs %v", i, got[i], want[i])
		}
	}
}

// TestAlphaMatters proves processing alpha is not decorative: on images
// whose alpha plane varies, dropping the channel (cn=3) produces a
// materially different response map than the full 4-channel computation
// OpenCV performs — which is exactly why Match only skips alpha when it is
// provably a zero contributor.
func TestAlphaMatters(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	iw, ih, tw, th := 160, 120, 32, 24
	img, tpl := randPix(iw*ih*4, rng), randPix(tw*th*4, rng)
	rw, rh := iw-tw+1, ih-th+1
	got3 := make([]float32, rw*rh)
	got4 := make([]float32, rw*rh)
	matchU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 3, 4, 2, got3)
	matchU8(img, iw*4, iw, ih, tpl, tw*4, tw, th, 4, 4, 2, got4)
	worst := 0.0
	for i := range got3 {
		if d := math.Abs(float64(got3[i]) - float64(got4[i])); d > worst {
			worst = d
		}
	}
	if worst < 1e-3 {
		t.Fatalf("varying alpha should change the map; cn=3 vs cn=4 worst diff only %g", worst)
	}
}

// TestAlphaMixedConstancy: the skip must not trigger when only ONE image has
// a constant alpha plane; Match must equal the full 4-channel reference.
func TestAlphaMixedConstancy(t *testing.T) {
	rng := rand.New(rand.NewSource(92))
	parent := image.NewRGBA(image.Rect(0, 0, 120, 90))
	copy(parent.Pix, randPix(len(parent.Pix), rng)) // varying alpha
	sub := image.NewRGBA(image.Rect(0, 0, 24, 18))
	for y := 0; y < 18; y++ {
		copy(sub.Pix[y*sub.Stride:y*sub.Stride+24*4], parent.Pix[(40+y)*parent.Stride+60*4:])
	}
	for i := 3; i < len(sub.Pix); i += 4 {
		sub.Pix[i] = 128 // sub constant, parent varying -> full path required
	}
	_, _, _, maxV, maxX, maxY := Match(parent, sub)
	want := refMatch(parent.Pix, parent.Stride, 120, 90, sub.Pix, sub.Stride, 24, 18, 4, 4)
	got := make([]float32, len(want))
	gMinV, _, _, gMaxV, gMaxX, gMaxY := matchU8(parent.Pix, parent.Stride, 120, 90, sub.Pix, sub.Stride, 24, 18, 4, 4, 2, got)
	_ = gMinV
	if maxV != gMaxV || maxX != gMaxX || maxY != gMaxY {
		t.Fatalf("Match took the skip despite mixed alpha constancy: (%d,%d) %v vs cn4 (%d,%d) %v",
			maxX, maxY, maxV, gMaxX, gMaxY, gMaxV)
	}
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); d > 1e-4 {
			t.Fatalf("mixed-constancy map deviates from 4-channel reference at %d: %v vs %v", i, got[i], want[i])
		}
	}
}

// TestFindAllOccurrences demonstrates and verifies the MatchMap recipe:
// the same template stamped at five known spots is recovered exactly by
// threshold + local-maximum suppression.
func TestFindAllOccurrences(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	parent := image.NewRGBA(image.Rect(0, 0, 400, 300))
	for i := 0; i < len(parent.Pix); i += 4 { // weak background noise
		parent.Pix[i] = uint8(rng.Intn(64))
		parent.Pix[i+1] = uint8(rng.Intn(64))
		parent.Pix[i+2] = uint8(rng.Intn(64))
		parent.Pix[i+3] = 255
	}
	tpl := image.NewRGBA(image.Rect(0, 0, 32, 24))
	copy(tpl.Pix, randPix(len(tpl.Pix), rng))
	for i := 3; i < len(tpl.Pix); i += 4 {
		tpl.Pix[i] = 255
	}
	spots := []image.Point{{10, 12}, {200, 40}, {350, 260}, {60, 240}, {180, 150}}
	for _, p := range spots {
		for y := 0; y < 24; y++ {
			copy(parent.Pix[(p.Y+y)*parent.Stride+p.X*4:(p.Y+y)*parent.Stride+(p.X+32)*4],
				tpl.Pix[y*tpl.Stride:])
		}
	}

	resp, w, h := MatchMap(parent, tpl)
	var hits []image.Point
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := resp[y*w+x]
			if v < 0.999 {
				continue
			}
			isMax := true // NMS over a template-sized neighborhood
			for dy := -12; dy <= 12 && isMax; dy++ {
				for dx := -16; dx <= 16; dx++ {
					nx, ny := x+dx, y+dy
					if nx >= 0 && ny >= 0 && nx < w && ny < h && resp[ny*w+nx] > v {
						isMax = false
						break
					}
				}
			}
			if isMax {
				hits = append(hits, image.Point{x, y})
			}
		}
	}
	if len(hits) != len(spots) {
		t.Fatalf("expected %d occurrences, found %d: %v", len(spots), len(hits), hits)
	}
	found := map[image.Point]bool{}
	for _, h := range hits {
		found[h] = true
	}
	for _, p := range spots {
		if !found[p] {
			t.Fatalf("planted occurrence at %v not found; hits: %v", p, hits)
		}
	}
}
