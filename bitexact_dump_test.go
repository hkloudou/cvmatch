package cvmatch

// TestDumpBitExactHashes is a development harness (not part of the normal
// suite): when CVMATCH_HASHDUMP is set, it hashes the full response map and
// extrema of both cores across a matrix of shapes, channel counts, strides
// and thread counts, and writes one line per case to the given file.
// Comparing the file before and after an optimization proves the change is
// bit-exact. Skipped unless the env var is set.

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestDumpBitExactHashes(t *testing.T) {
	path := os.Getenv("CVMATCH_HASHDUMP")
	if path == "" {
		t.Skip("CVMATCH_HASHDUMP not set")
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	type tc struct{ iw, ih, tw, th, cn, step, ipad, tpad int }
	cases := []tc{
		{97, 61, 17, 13, 1, 1, 0, 0},
		{97, 61, 17, 13, 4, 4, 0, 0},
		{120, 90, 24, 18, 3, 4, 0, 0},
		{300, 200, 31, 29, 1, 1, 7, 3}, // strided
		{64, 64, 64, 64, 1, 1, 0, 0},   // template == image
		{50, 40, 1, 1, 4, 4, 0, 0},     // 1x1 template
		{640, 400, 96, 32, 3, 4, 0, 0}, // multi-block tiling
		{700, 500, 96, 64, 3, 4, 0, 0}, // several tiles/bands
		{40, 300, 33, 257, 1, 1, 0, 0}, // tall template, tiny rw
		{1280, 720, 96, 96, 3, 4, 0, 0},
		{1280, 720, 96, 96, 4, 4, 0, 0},
		{513, 511, 33, 31, 2, 2, 5, 2}, // odd sizes, 2ch, strided
		{256, 256, 255, 255, 1, 1, 0, 0},
		{2000, 100, 64, 8, 1, 1, 0, 0}, // wide and flat
	}
	for ci, c := range cases {
		rng := rand.New(rand.NewSource(int64(1000 + ci)))
		istride, tstride := (c.iw+c.ipad)*c.step, (c.tw+c.tpad)*c.step
		img := randPix(istride*c.ih, rng)
		tpl := randPix(tstride*c.th, rng)
		rw, rh := c.iw-c.tw+1, c.ih-c.th+1
		for _, threads := range []int{1, 2, 4, 16} {
			for _, core := range eachCore() {
				res := make([]float32, rw*rh)
				minV, minX, minY, maxV, maxX, maxY := core.fn(img, istride, c.iw, c.ih, tpl, tstride, c.tw, c.th, c.cn, c.step, threads, res)
				h := sha256.New()
				buf := make([]byte, 4)
				for _, v := range res {
					binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
					h.Write(buf)
				}
				fmt.Fprintf(f, "case=%02d threads=%02d core=%-18s map=%x min=%08x@%d,%d max=%08x@%d,%d\n",
					ci, threads, core.name, h.Sum(nil)[:12],
					math.Float32bits(minV), minX, minY, math.Float32bits(maxV), maxX, maxY)
			}
		}
	}
}
