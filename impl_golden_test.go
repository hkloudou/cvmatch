package cvmatch

import (
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"
)

// Golden re-record mode: never edit hashes by hand — `make regolden
// REASON="..."` runs the native tolerance parity first, then rewrites
// golden_hashes_test.go through these flags.
var (
	recordGoldens = flag.Bool("cvmatch.record", false, "re-record golden_hashes_test.go")
	recordReason  = flag.String("cvmatch.reason", "", "reason for the re-record (required)")
)

// goldenHash folds a response map and its extrema into one FNV-1a 64-bit
// value over the exact float32 bit patterns (little-endian), so a single
// constant pins every output bit of a match call.
func goldenHash(res []float32, minV float32, minX, minY int, maxV float32, maxX, maxY int) uint64 {
	h := fnv.New64a()
	var b [4]byte
	put := func(bits uint32) {
		b[0], b[1], b[2], b[3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
		h.Write(b[:])
	}
	for _, v := range res {
		put(math.Float32bits(v))
	}
	put(math.Float32bits(minV))
	put(uint32(minX))
	put(uint32(minY))
	put(math.Float32bits(maxV))
	put(uint32(maxX))
	put(uint32(maxY))
	return h.Sum64()
}

// goldenCases pins one deterministic input per dispatch shape; the recorded
// hashes live in golden_hashes_test.go. Seed 110 is the flat-window guard
// canary: an almost-uniform image where nearly every window has zero
// variance, so the normalization guard ladder (den==0, the lim clamp, the
// ±1 saturation band) is the code under test — the canary keeps guard
// behavior pinned while FFT restructures spend tolerance budget.
type goldenCase struct {
	seed           int64
	name           string
	iw, ih, tw, th int
	cn, step       int
	ipad, tpad     int // extra row padding in pixels (strided sub-image case)
}

var goldenCases = []goldenCase{
	{101, "grayscale", 97, 61, 17, 13, 1, 1, 0, 0},
	{102, "full RGBA", 97, 61, 17, 13, 4, 4, 0, 0},
	{103, "RGBA with alpha skipped (cn=3, step=4)", 120, 90, 24, 18, 3, 4, 0, 0},
	{104, "strided rows (sub-image)", 300, 200, 31, 29, 1, 1, 7, 3},
	{105, "template == image (1x1 result)", 64, 64, 64, 64, 1, 1, 0, 0},
	{106, "1x1 template", 50, 40, 1, 1, 4, 4, 0, 0},
	{107, "multi-block FFT tiling", 640, 400, 96, 32, 3, 4, 0, 0},
	{108, "packed RGB (step==cn=3, scalar pack fallback)", 128, 96, 5, 4, 3, 3, 0, 0},
	{109, "tall template, tiny result width", 40, 300, 33, 257, 1, 1, 0, 0},
	{110, "flat-window guard canary", 200, 150, 32, 32, 1, 1, 0, 0},
}

func goldenInput(c goldenCase) (img, tpl []uint8, istride, tstride int) {
	istride, tstride = (c.iw+c.ipad)*c.step, (c.tw+c.tpad)*c.step
	if c.seed == 110 {
		img = make([]uint8, istride*c.ih)
		tpl = make([]uint8, tstride*c.th)
		for i := range img {
			img[i] = 128
		}
		for i := range tpl {
			tpl[i] = 128
		}
		img[75*istride+100] = 129 // one perturbed pixel in the parent
		tpl[5*tstride+7] = 129    // template not exactly flat (tiny templNorm)
		return img, tpl, istride, tstride
	}
	rng := rand.New(rand.NewSource(c.seed))
	return randPix(istride*c.ih, rng), randPix(tstride*c.th, rng), istride, tstride
}

// TestGoldenOutputs asserts the core reproduces the recorded output bits —
// map and extrema — for every dispatch shape, single- and multi-threaded.
// The same constants must hold on every architecture and in both build
// modes (SIMD kernels and the scalar fallbacks execute identical op
// sequences). With -cvmatch.record it instead rewrites the recorded table
// (only via `make regolden`, which gates on native tolerance parity first).
func TestGoldenOutputs(t *testing.T) {
	if len(goldenHashes) != len(goldenCases) && !*recordGoldens {
		t.Fatalf("golden_hashes_test.go has %d hashes for %d cases — run make regolden",
			len(goldenHashes), len(goldenCases))
	}
	got := make([]uint64, len(goldenCases))
	for ci, c := range goldenCases {
		img, tpl, istride, tstride := goldenInput(c)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1
		for _, nt := range []int{1, 4} {
			res := make([]float32, rw*rh)
			minV, minX, minY, maxV, maxX, maxY := matchU8(img, istride, c.iw, c.ih,
				tpl, tstride, c.tw, c.th, c.cn, c.step, nt, res)
			h := goldenHash(res, minV, minX, minY, maxV, maxX, maxY)
			if nt == 1 {
				got[ci] = h
			} else if h != got[ci] {
				t.Fatalf("case %d (%s): threads=4 hash %#016x != threads=1 %#016x — determinism broken, never record this",
					ci, c.name, h, got[ci])
			}
			if !*recordGoldens && h != goldenHashes[ci] {
				t.Errorf("case %d (%s, threads=%d): output hash %#016x, want recorded %#016x\n"+
					"output bits changed vs recorded goldens; if this is an intended arithmetic-sequence\n"+
					"change run `make regolden REASON=...` (never paste hashes by hand) — otherwise this\n"+
					"is a determinism regression", ci, c.name, nt, h, goldenHashes[ci])
			}
		}
	}
	if *recordGoldens {
		if t.Failed() {
			t.Fatal("refusing to record: determinism check failed")
		}
		writeGoldens(t, got)
	}
}

func writeGoldens(t *testing.T, hashes []uint64) {
	if strings.TrimSpace(*recordReason) == "" {
		t.Fatal("recording requires -cvmatch.reason (use make regolden REASON=\"...\")")
	}
	const path = "golden_hashes_test.go"
	old, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	changed := len(hashes) != len(goldenHashes)
	for i := range goldenHashes {
		if hashes[i] != goldenHashes[i] {
			changed = true
		}
	}
	if m := regexp.MustCompile(`Reason: (.*)`).FindSubmatch(old); m != nil &&
		changed && strings.TrimSpace(string(m[1])) == strings.TrimSpace(*recordReason) {
		t.Fatal("hashes changed but REASON equals the currently recorded one — write a fresh reason")
	}
	if !changed {
		t.Logf("goldens unchanged; refreshing header only")
	}
	var b strings.Builder
	b.WriteString("package cvmatch\n\n")
	b.WriteString("// Code generated by `make regolden REASON=\"...\"`. DO NOT EDIT BY HAND.\n//\n")
	fmt.Fprintf(&b, "// Recorded: %s — Reason: %s\n//\n",
		time.Now().UTC().Format("2006-01-02"), strings.TrimSpace(*recordReason))
	b.WriteString(`// One constant per goldenCases entry, in order. These pin the
// implementation's own deterministic output (self-identity across
// architectures, build modes and thread counts); validity vs OpenCV is
// established by the bench tolerance parity gates at record time.
var goldenHashes = [...]uint64{
`)
	for i, h := range hashes {
		fmt.Fprintf(&b, "\t%#016x, // %d %s\n", h, goldenCases[i].seed, goldenCases[i].name)
	}
	b.WriteString("}\n")
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	t.Logf("recorded %d golden hashes (reason: %s)", len(hashes), *recordReason)
}
