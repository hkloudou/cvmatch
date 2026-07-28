package cvmatch

import (
	"image"
	"testing"
)

func mkM(w, h int, flat bool) *Matcher {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		if flat {
			img.Pix[i] = 100
		} else {
			img.Pix[i] = uint8(i*13 + i/7)
		}
	}
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 255
	}
	return NewMatcher(img)
}

// Orthogonal extremes must never synthesize a giant virtual plan: a
// wide×short and a tall×narrow member get separate buckets whose
// dimensions are their own; dominated shapes fold into the dominating
// bucket; flat members shape nothing.
func TestBucketize(t *testing.T) {
	wide, tall := mkM(300, 8, false), mkM(8, 300, false)
	small, mid := mkM(24, 24, false), mkM(200, 100, false)
	flat := mkM(400, 300, true)

	bs := bucketize([]*Matcher{wide, tall})
	if len(bs) != 2 {
		t.Fatalf("orthogonal shapes: %d buckets, want 2", len(bs))
	}
	for _, b := range bs {
		if b.tw > 300 || b.th > 300 || (b.tw == 300 && b.th == 300) {
			t.Fatalf("synthesized bucket %dx%d", b.tw, b.th)
		}
	}

	bs = bucketize([]*Matcher{mid, small}) // 256x128 class dominates 32x32
	if len(bs) != 1 || bs[0].tw != 200 || bs[0].th != 100 {
		t.Fatalf("dominance merge: %+v", bs)
	}

	bs = bucketize([]*Matcher{flat, small})
	if len(bs) != 1 || bs[0].tw != 24 || bs[0].th != 24 {
		t.Fatalf("flat member shaped a bucket: %+v", bs)
	}
}
