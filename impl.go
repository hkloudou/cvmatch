package cvmatch

// The cvmatch core: block-FFT cross-correlation with OpenCV's tile
// heuristic, sliding-column-sum normalization with OpenCV's exact guards,
// fused minMaxLoc, and tile/band parallelism with bit-identical output for
// any worker count. Pure Go with SIMD kernels (internal/simd) on the hot
// loops; TestGoldenOutputs pins its output bits to values cross-validated
// against OpenCV.

import (
	"fmt"
	"math"
	"math/bits"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/hkloudou/cvmatch/internal/simd"
)

const maxThreads = 16

// Scratch pools: every pooled buffer is fully overwritten before use
// (block spectra include their zero padding, column sums are cleared in
// colBuildGo, the internal result map is stored before it is read), so
// dirty reuse is safe and steady-state matching allocates nothing.
//
// Peak-RSS note: the benchmarked memory metric (VmHWM) is a monotone
// high-water mark — trimming pools, forcing GC or FreeOSMemory can never
// lower it; only reducing the peak of concurrently-live bytes can.
// sync.Pool's victim cache already sheds idle retention on its own.
type slicePool[T any] struct{ p sync.Pool }

func (sp *slicePool[T]) get(n int) []T {
	if v, _ := sp.p.Get().(*[]T); v != nil && cap(*v) >= n {
		return (*v)[:n]
	}
	return make([]T, n)
}

func (sp *slicePool[T]) put(s []T) { sp.p.Put(&s) }

var (
	cplxPool slicePool[complex64]
	f32Pool  slicePool[float32]
	f64Pool  slicePool[float64]
	i32Pool  slicePool[int32]
	i64Pool  slicePool[int64]
	bytePool slicePool[uint8]
)

func nextPow2(v int) int {
	n := 1
	for n < v {
		n <<= 1
	}
	return n
}

func clampThreads(n int) int {
	if n < 1 {
		return 1
	}
	if n > maxThreads {
		return maxThreads
	}
	return n
}

// runParallel executes fn(0..n-1) on n goroutines. Work is statically
// partitioned by index, so results never depend on scheduling.
func runParallel(n int, fn func(w int)) {
	if n <= 1 {
		fn(0)
		return
	}
	var wg sync.WaitGroup
	for i := 1; i < n; i++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			fn(w)
		}(i)
	}
	fn(0)
	wg.Wait()
}

// ------------------------------------------------------------------- FFT --

// sincospiFrac returns cos, sin of pi*j/half (0 <= j < half, half a power
// of two): exact dyadic range reduction plus fdlibm-style minimax kernels
// in plain sequenced double ops — no library trig, so the twiddle tables
// (and therefore the outputs) are bit-identical on every platform
// regardless of the system libm. The float64 conversions around products
// pin each op to one rounding (Go may otherwise fuse mul-adds on some
// architectures).
func sincospiFrac(j, half int) (float32, float32) {
	m := 2 * j    // pi*j/half = k*(pi/2) + u*(pi/2), u in [0,1)
	k := m / half // 0 or 1
	rem := m - k*half
	u := float64(rem) / float64(half) // exact: half is a power of two
	swap := false
	if u > 0.5 { // co-function fold, exact dyadic subtraction
		u = 1.0 - u
		swap = true
	}
	y := float64(u * 1.57079632679489661923) // u*(pi/2), one rounding
	z := float64(y * y)
	r := float64(z * (4.16666666666666019037e-02 +
		float64(z*(-1.38888888888741095749e-03+
			float64(z*(2.48015872894767294178e-05+
				float64(z*(-2.75573143513906633035e-07+
					float64(z*(2.08757232129817482790e-09+
						float64(z*-1.13596475577881948265e-11)))))))))))
	hz := float64(0.5 * z)
	w := 1.0 - hz
	c := w + (((1.0 - w) - hz) + float64(z*r))
	r2 := 8.33333333332248946124e-03 +
		float64(z*(-1.98412698298579493134e-04+
			float64(z*(2.75573137070700676789e-06+
				float64(z*(-2.50507602534068634195e-08+
					float64(z*1.58969099521155010221e-10)))))))
	v := float64(z * y)
	s := y + float64(v*(-1.66666666666666324348e-01+float64(z*r2)))
	if swap {
		c, s = s, c
	}
	if k != 0 { // rotate by pi/2
		c, s = -s, c
	}
	return float32(c), float32(s)
}

// makeSwapPairs lists the bit-reversal swaps (i, br[i]) with i < br[i];
// iterating pairs avoids the branchy per-element compare. Pure data
// movement — results are unaffected.
func makeSwapPairs(n int) []int32 {
	pairs := make([]int32, 0, n)
	br := 0
	for i := 0; i < n; i++ {
		if br > i {
			pairs = append(pairs, int32(i), int32(br))
		}
		bit := n >> 1
		for br&bit != 0 {
			br ^= bit
			bit >>= 1
		}
		br |= bit
	}
	return pairs
}

// fftTables caches the per-size twiddle/swap-pair tables (immutable once
// built; real workloads cycle through a handful of power-of-two sizes).
// Sizes above 4096 stay per-call to bound the cache.
var (
	fftTabMu sync.Mutex
	fftTabs  = map[int]*fftTab{}
)

type fftTab struct {
	n     int
	triT  []complex64 // radix-4 stage triplets; lazily built on uncached sizes
	pairs []int32
}

// tri returns the radix-4 triplet table. Cached sizes build it eagerly
// under the cache lock (immutable once published, safe to share); sizes
// above 4096 return per-call structs that fill on first read — plans
// are built on a single goroutine before any fan-out, so the lazy fill
// needs no lock.
func (t *fftTab) tri() []complex64 {
	if t.triT == nil {
		t.triT = makeTriTwiddles(t.n)
	}
	return t.triT
}

func fftTables(n int) *fftTab {
	if n > 4096 {
		return &fftTab{n: n, pairs: makeSwapPairs(n)}
	}
	fftTabMu.Lock()
	t := fftTabs[n]
	if t == nil {
		t = &fftTab{n: n, triT: makeTriTwiddles(n), pairs: makeSwapPairs(n)}
		fftTabs[n] = t
	}
	fftTabMu.Unlock()
	return t
}

// colRange runs fn over the whole width, or splits it across a worker team
// (runParallel doubles as the barrier between column-FFT passes).
func colRange(team, width int, fn func(c0, c1 int)) {
	if team <= 1 {
		fn(0, width)
		return
	}
	runParallel(team, func(u int) {
		c0, c1 := u*width/team, (u+1)*width/team
		if c0 < c1 {
			fn(c0, c1)
		}
	})
}

// -------------------------------------------------------- FFT plan/blocks --

type goPlan struct {
	dftW, dftH, hw int
	blockW, blockH int
	triW, triH     []complex64 // radix-4 stage triplets (dftW rows, dftH columns)
	prW, prH       []int32
}

// newGoPlan picks the DFT tile geometry by an integer cost-model argmin
// over power-of-two (dftW, dftH) pairs — under the Phase 7 contract the
// geometry only has to be fast, not to reproduce OpenCV's crossCorr
// block heuristic. Each candidate is scored with a closed-form model of
// this pipeline's own kernels (validated within ~10% against interleaved
// A/Bs): packed row FFTs at 5·n·log2 n per pair, two full column passes,
// and a linear conj-multiply/pack/emit term, summed over the real tile
// grid including the short edge bands. The search is pure integer
// arithmetic (bits.Len, int64), so every platform and build picks the
// identical plan; ties go to the smaller spectrum.
func newGoPlan(tw, th, rw, rh int) *goPlan {
	const areaCap = 1 << 22 // bounds spec at ~16.8 MB; searched optima stay far below
	minW, minH := max(2, nextPow2(tw)), max(2, nextPow2(th))
	maxW, maxH := max(minW, nextPow2(rw+tw-1)), max(minH, nextPow2(rh+th-1))
	log2 := func(n int) int64 { return int64(bits.Len(uint(n)) - 1) }
	bestW, bestH := minW, minH
	var bestCost, bestArea int64 = -1, 0
	for dftW := minW; dftW <= maxW; dftW <<= 1 {
		for dftH := minH; dftH <= maxH; dftH <<= 1 {
			area := int64(dftW) * int64(dftH)
			if area > areaCap {
				continue
			}
			blockW := min(dftW-tw+1, rw)
			blockH := min(dftH-th+1, rh)
			ntx := int64((rw + blockW - 1) / blockW)
			hw := dftW/2 + 1
			rowFFT := 5 * int64(dftW) * log2(dftW)
			colPass := 5 * int64(dftH) * log2(dftH) * int64(hw)
			specN := int64(dftH) * int64(hw)
			var cost int64
			for y0 := 0; y0 < rh; y0 += blockH {
				bh := min(blockH, rh-y0)
				loadH := min(bh+th-1, dftH) // zero pad rows are skipped by the forward pass
				cost += ntx * (int64((loadH+1)/2+(bh+1)/2)*rowFFT + 2*colPass + 16*specN)
			}
			if bestCost < 0 || cost < bestCost || (cost == bestCost && area < bestArea) {
				bestCost, bestArea, bestW, bestH = cost, area, dftW, dftH
			}
		}
	}
	dftW, dftH := bestW, bestH
	p := &goPlan{dftW: dftW, dftH: dftH, hw: dftW/2 + 1}
	p.blockW = dftW - tw + 1
	if p.blockW > rw {
		p.blockW = rw
	}
	p.blockH = dftH - th + 1
	if p.blockH > rh {
		p.blockH = rh
	}
	tabW := fftTables(dftW)
	p.triW, p.prW = tabW.tri(), tabW.pairs
	tabH := tabW
	if dftH != dftW {
		tabH = fftTables(dftH)
	}
	p.triH, p.prH = tabH.tri(), tabH.pairs
	return p
}

// forwardRowPair packs image rows r,r+1 into one complex row FFT and
// untangles the two spectra into spec rows r,r+1. Row pairs touch disjoint
// spec rows and share no state beyond the per-worker scratch z, so they
// may run in any order or concurrently with bit-identical results.
func forwardRowPair(chanBase []uint8, stride, step, x0, y0, loadW, loadH int, p *goPlan, spec, z []complex64, r int) {
	n, hw, mask := p.dftW, p.hw, p.dftW-1
	// The pack kernels implement the strides the public API produces (1 and
	// 4); other layouts (e.g. packed RGB, step 3) take the scalar loops.
	usePack := simd.Enabled && (step == 1 || step == 4)
	sa := spec[r*hw : r*hw+hw]
	sb := spec[(r+1)*hw : (r+1)*hw+hw]
	ra := chanBase[(y0+r)*stride+x0*step:]
	if r+1 < loadH {
		rb := chanBase[(y0+r+1)*stride+x0*step:]
		if usePack {
			simd.PackRows2(z[:loadW], ra, rb, step)
		} else {
			for x := 0; x < loadW; x++ {
				z[x] = complex(float32(ra[x*step]), float32(rb[x*step]))
			}
		}
	} else if usePack {
		simd.PackRows1(z[:loadW], ra, step)
	} else {
		for x := 0; x < loadW; x++ {
			z[x] = complex(float32(ra[x*step]), 0)
		}
	}
	clear(z[loadW:n])
	fftR4(z, p.triW, p.prW, false)
	if simd.Enabled {
		// k = 0 wraps to itself ((n-0)&mask == 0); the kernel covers
		// the wrap-free k >= 1 range with the same op sequence.
		zk := z[0]
		sa[0] = complex(0.5*(real(zk)+real(zk)), 0.5*(imag(zk)-imag(zk)))
		sb[0] = complex(0.5*(imag(zk)+imag(zk)), 0.5*(real(zk)-real(zk)))
		simd.Untangle(sa, sb, z, n, 1, hw)
	} else {
		for k := 0; k < hw; k++ {
			zk, zn := z[k], z[(n-k)&mask]
			sa[k] = complex(0.5*(real(zk)+real(zn)), 0.5*(imag(zk)-imag(zn)))
			sb[k] = complex(0.5*(imag(zk)+imag(zn)), 0.5*(real(zn)-real(zk)))
		}
	}
}

// blockForwardGo: real 2D forward DFT of one channel of a uint8 block, two
// real rows packed per complex FFT. With team > 1 the independent row
// pairs stride across a worker team (spare threads on single-tile scenes)
// — pure scheduling, bit-identical output.
func blockForwardGo(chanBase []uint8, stride, step, x0, y0, loadW, loadH int, p *goPlan, spec, z []complex64, team int) {
	hw := p.hw
	loaded := min(p.dftH, (loadH+1)&^1)
	clear(spec[loaded*hw : p.dftH*hw])
	if team <= 1 {
		for r := 0; r < loadH && r < p.dftH; r += 2 {
			forwardRowPair(chanBase, stride, step, x0, y0, loadW, loadH, p, spec, z, r)
		}
	} else {
		runParallel(team, func(w int) {
			zw := z
			if w > 0 {
				zw = cplxPool.get(p.dftW)
				defer cplxPool.put(zw)
			}
			for r := 2 * w; r < loadH && r < p.dftH; r += 2 * team {
				forwardRowPair(chanBase, stride, step, x0, y0, loadW, loadH, p, spec, zw, r)
			}
		})
	}
	colsR4Go(spec, p.dftH, hw, p.triH, p.prH, false, z, team)
}

func mulConjGo(spec, tspec []complex64) {
	if simd.Enabled {
		simd.MulConj(spec, tspec)
		return
	}
	for i, a := range spec {
		b := tspec[i]
		// Explicit float32 roundings, matching the SIMD kernel (see bfly).
		re := float32(real(a)*real(b)) + float32(imag(a)*imag(b))
		im := float32(imag(a)*real(b)) - float32(real(a)*imag(b))
		spec[i] = complex(re, im)
	}
}

// inverseRowPair combines spec rows r,r+1 into one inverse row FFT and
// emits result rows y0+r, y0+r+1. Row pairs write disjoint result rows and
// share nothing beyond the per-worker scratch z — concurrency-safe with
// bit-identical output, exactly like forwardRowPair.
func inverseRowPair(p *goPlan, spec, z []complex64, res []float32, rw, x0, y0, bw, bh int, add bool, r int) {
	n, hw := p.dftW, p.hw
	sa := spec[r*hw : r*hw+hw]
	sb := spec[(r+1)*hw : (r+1)*hw+hw]
	if simd.Enabled {
		simd.CombineLow(z[:hw], sa, sb)
		simd.CombineHigh(z, sa, sb, n, hw)
	} else {
		for k := 0; k < hw; k++ {
			z[k] = complex(real(sa[k])-imag(sb[k]), imag(sa[k])+real(sb[k]))
		}
		for k := hw; k < n; k++ {
			m := n - k
			z[k] = complex(real(sa[m])+imag(sb[m]), real(sb[m])-imag(sa[m]))
		}
	}
	fftR4(z, p.triW, p.prW, true)
	o := res[(y0+r)*rw+x0:]
	var o2 []float32
	if r+1 < bh {
		o2 = res[(y0+r+1)*rw+x0:]
	}
	switch {
	case simd.Enabled:
		simd.EmitRe(o[:bw], z, add)
		if o2 != nil {
			simd.EmitIm(o2[:bw], z, add)
		}
	case add:
		for x := 0; x < bw; x++ {
			o[x] += real(z[x])
		}
		if o2 != nil {
			for x := 0; x < bw; x++ {
				o2[x] += imag(z[x])
			}
		}
	default:
		for x := 0; x < bw; x++ {
			o[x] = real(z[x])
		}
		if o2 != nil {
			for x := 0; x < bw; x++ {
				o2[x] = imag(z[x])
			}
		}
	}
}

func blockInverseEmitGo(p *goPlan, spec, z []complex64, res []float32, rw, x0, y0, bw, bh int, add bool, team int) {
	colsR4Go(spec, p.dftH, p.hw, p.triH, p.prH, true, z, team)
	if team <= 1 {
		for r := 0; r < bh; r += 2 {
			inverseRowPair(p, spec, z, res, rw, x0, y0, bw, bh, add, r)
		}
		return
	}
	runParallel(team, func(w int) {
		zw := z
		if w > 0 {
			zw = cplxPool.get(p.dftW)
			defer cplxPool.put(zw)
		}
		for r := 2 * w; r < bh; r += 2 * team {
			inverseRowPair(p, spec, zw, res, rw, x0, y0, bw, bh, add, r)
		}
	})
}

// crossCorrGo runs the tile-parallel raw cross-correlation. Each tile owns a
// disjoint result region and runs every channel in order, so per-element
// arithmetic is identical for any worker count.
func crossCorrGo(img []uint8, istride int, tpl []uint8, tstride, step, cn, tw, th, rw, rh, threads int, p *goPlan, result []float32) {
	specN := p.dftH * p.hw
	// Tiny blocks cannot amortize a per-pass fan-out (a 20x20 match
	// regressed ~3x when it fanned out unconditionally — codex finding on
	// PR #13), so team parallelism only engages when the spectrum carries
	// real work.
	teamOK := specN >= 1<<16
	tspec := cplxPool.get(cn * specN)
	// Template spectra: the channels are disjoint, and within one channel
	// the team path and the chunked scale partition elementwise work — all
	// pure scheduling, per-element arithmetic unchanged.
	tcw := min(threads, cn)
	tteam := 1
	if teamOK {
		tteam = threads / tcw
	}
	runParallel(tcw, func(w int) {
		z := cplxPool.get(p.dftW)
		defer cplxPool.put(z)
		scale := 1 / (float32(p.dftW) * float32(p.dftH))
		for k := w; k < cn; k += tcw {
			ts := tspec[k*specN : (k+1)*specN]
			blockForwardGo(tpl[k:], tstride, step, 0, 0, tw, th, p, ts, z, tteam)
			runParallel(tteam, func(u int) {
				lo, hi := u*specN/tteam, (u+1)*specN/tteam
				if simd.Enabled {
					simd.ScaleCplx(ts[lo:hi], scale)
					return
				}
				for i := lo; i < hi; i++ {
					ts[i] = complex(real(ts[i])*scale, imag(ts[i])*scale)
				}
			})
		}
	})
	ntx := (rw + p.blockW - 1) / p.blockW
	nty := (rh + p.blockH - 1) / p.blockH
	ntiles := ntx * nty
	nw := threads
	if nw > ntiles {
		nw = ntiles
	}
	// Spare workers (threads beyond the tile count — always the case on
	// single-tile scenes) form a per-tile team that strides the independent
	// row-pair and elementwise phases inside each block. Pure scheduling of
	// disjoint work: per-element arithmetic and channel order are unchanged,
	// so output stays bit-identical for every (threads, ntiles) combination.
	team := 1
	if teamOK {
		team = threads / nw
	}
	// Tiles are claimed from a shared queue, longest first (ragged edge
	// tiles load less data), so no worker sits on a full tile while the
	// rest idle out — tiles own disjoint result regions, so claim order
	// cannot affect output.
	order := i32Pool.get(ntiles)
	for t := range order {
		order[t] = int32(t)
	}
	if nw > 1 {
		sort.SliceStable(order, func(a, b int) bool {
			ta, tb := int(order[a]), int(order[b])
			ca := int64(min(p.blockW, rw-(ta%ntx)*p.blockW)+tw-1) * int64(min(p.blockH, rh-(ta/ntx)*p.blockH)+th-1)
			cb := int64(min(p.blockW, rw-(tb%ntx)*p.blockW)+tw-1) * int64(min(p.blockH, rh-(tb/ntx)*p.blockH)+th-1)
			return ca > cb
		})
	}
	// A short last row band whose loaded rows fit a smaller power of two
	// gets its own transform height: same dftW and band geometry, half (or
	// less) the column-FFT work on that band. The 7.1 argmin already
	// absorbs edge waste into the uniform plan on most shapes — this
	// catches the residual (two-band plans with a short tail, ~18% of the
	// correlation model on the affected published scenes). The shrunk
	// template spectrum costs no new transforms: the template's padded
	// support (th+1 <= dftH2 rows) makes the decimation identity
	// X_dftH[k<<s] = X_dftH2[k] exact per column, so it is the stride-2^s
	// row gather of tspec times 2^s — an exact power-of-two multiply. The
	// band's tiles run one fixed op sequence at dftH2, so output stays
	// deterministic everywhere; scores move only within the tolerance
	// contract (goldens re-recorded, parity gates prove the budget).
	p2, tspec2 := p, tspec
	lastY0 := ((rh - 1) / p.blockH) * p.blockH
	if lastBh := rh - lastY0; lastY0 > 0 {
		if dh2 := max(2, nextPow2(lastBh+th-1)); dh2 < p.dftH {
			q := *p
			q.dftH = dh2
			tabH := fftTables(dh2)
			q.triH, q.prH = tabH.tri(), tabH.pairs
			p2 = &q
			tspec2 = cplxPool.get(cn * dh2 * p.hw)
			defer cplxPool.put(tspec2)
			shift := bits.TrailingZeros(uint(p.dftH)) - bits.TrailingZeros(uint(dh2))
			up := float32(int32(1) << shift)
			for k := 0; k < cn; k++ {
				for r := 0; r < dh2; r++ {
					src := tspec[k*specN+(r<<shift)*p.hw:][:p.hw]
					dst := tspec2[k*dh2*p.hw+r*p.hw:][:p.hw]
					for x, v := range src {
						dst[x] = complex(real(v)*up, imag(v)*up)
					}
				}
			}
		}
	}
	var next atomic.Int64
	runParallel(nw, func(w int) {
		spec := cplxPool.get(specN)
		z := cplxPool.get(p.dftW)
		for {
			ti := next.Add(1) - 1
			if ti >= int64(ntiles) {
				break
			}
			t := int(order[ti])
			x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
			bw, bh := min(p.blockW, rw-x0), min(p.blockH, rh-y0)
			bp, bt := p, tspec
			if y0 == lastY0 && p2 != p {
				bp, bt = p2, tspec2
			}
			sn := bp.dftH * bp.hw
			for k := 0; k < cn; k++ {
				blockForwardGo(img[k:], istride, step, x0, y0, bw+tw-1, bh+th-1, bp, spec, z, team)
				ts := bt[k*sn : (k+1)*sn]
				if team <= 1 {
					mulConjGo(spec[:sn], ts)
				} else {
					runParallel(team, func(u int) {
						lo := u * sn / team
						hi := (u + 1) * sn / team
						mulConjGo(spec[lo:hi], ts[lo:hi])
					})
				}
				blockInverseEmitGo(bp, spec, z, result, rw, x0, y0, bw, bh, k > 0, team)
			}
		}
		cplxPool.put(spec)
		cplxPool.put(z)
	})
	i32Pool.put(order)
	cplxPool.put(tspec)
}

// ------------------------------------------------- normalization + scan --

type goExtrema struct {
	minV, maxV             float32
	minX, minY, maxX, maxY int
}

// Column-sum helpers: the sliding statistics are integer-valued (colSum <=
// 255*th, colSum2 <= 255^2*th*cn), so int32/int64 accumulation feeds the
// double window math bit-identical inputs while avoiding per-byte float
// conversions. The (step,cn) pairs the public API produces get unrolled
// loops; anything else takes the generic path. With cn==3, step==4 the
// alpha lane is accumulated too (stride stays 4) but never read.
func colBuildGo(colSum []int32, colSum2 []int64, img []uint8, istride, iw, cn, cs, step, y0, th int) {
	for i := range colSum {
		colSum[i] = 0
	}
	for i := range colSum2 {
		colSum2[i] = 0
	}
	// (A build-as-slide-from-zero-row variant was measured: it deletes the
	// switch below but costs ~2x integer work per built row and showed +9%
	// e2e on purego 1T gray — the dedicated loops stay.)
	u4 := cs == 4 && step == 4
	for y := y0; y < y0+th; y++ {
		row := img[y*istride:]
		switch {
		case step == 1 && cn == 1:
			row = row[:iw]
			for x, v := range row {
				colSum[x] += int32(v)
				colSum2[x] += int64(int32(v) * int32(v))
			}
		case u4:
			row = row[:iw*4]
			for i, v := range row {
				colSum[i] += int32(v)
			}
			if cn == 3 {
				for x := 0; x < iw; x++ {
					r := int32(row[x*4])
					g := int32(row[x*4+1])
					b := int32(row[x*4+2])
					colSum2[x] += int64(r*r + g*g + b*b)
				}
			} else {
				for x := 0; x < iw; x++ {
					r := int32(row[x*4])
					g := int32(row[x*4+1])
					b := int32(row[x*4+2])
					a := int32(row[x*4+3])
					colSum2[x] += int64(r*r + g*g + b*b + a*a)
				}
			}
		default:
			for x := 0; x < iw; x++ {
				for k := 0; k < cn; k++ {
					v := int32(row[x*step+k])
					colSum[x*cs+k] += v
					colSum2[x] += int64(v * v)
				}
			}
		}
	}
}

func colSlideGo(colSum []int32, colSum2 []int64, rsub, radd []uint8, iw, cn, cs, step int) {
	xv := 0
	if simd.Enabled && iw >= 8 {
		xv = iw &^ 7
	}
	switch {
	case step == 1 && cn == 1:
		if xv > 0 {
			simd.SlideCols1(colSum[:xv], colSum2[:xv], rsub, radd)
		}
		for x := xv; x < iw; x++ {
			a, b := int32(radd[x]), int32(rsub[x])
			colSum[x] += a - b
			colSum2[x] += int64(a*a - b*b)
		}
	case cs == 4 && step == 4:
		if xv > 0 {
			simd.SlideCols4(colSum[:xv*4], colSum2[:xv], rsub, radd, cn)
		}
		for i := xv * 4; i < iw*4; i++ {
			colSum[i] += int32(radd[i]) - int32(rsub[i])
		}
		if cn == 3 {
			for x := xv; x < iw; x++ {
				ar, ag, ab := int32(radd[x*4]), int32(radd[x*4+1]), int32(radd[x*4+2])
				br, bg, bb := int32(rsub[x*4]), int32(rsub[x*4+1]), int32(rsub[x*4+2])
				colSum2[x] += int64(ar*ar + ag*ag + ab*ab - br*br - bg*bg - bb*bb)
			}
		} else {
			for x := xv; x < iw; x++ {
				ar, ag, ab, aa := int32(radd[x*4]), int32(radd[x*4+1]), int32(radd[x*4+2]), int32(radd[x*4+3])
				br, bg, bb, ba := int32(rsub[x*4]), int32(rsub[x*4+1]), int32(rsub[x*4+2]), int32(rsub[x*4+3])
				colSum2[x] += int64(ar*ar + ag*ag + ab*ab + aa*aa - br*br - bg*bg - bb*bb - ba*ba)
			}
		}
	default:
		for x := 0; x < iw; x++ {
			for k := 0; k < cn; k++ {
				a, b := int32(radd[x*step+k]), int32(rsub[x*step+k])
				colSum[x*cs+k] += a - b
				colSum2[x] += int64(a*a - b*b)
			}
		}
	}
}

// normChunkGo is the result-row chunk for the normalize scan: window sums
// are spilled per chunk so the buffer stays L1-resident.
const normChunkGo = 256

// abs32 clears the sign bit — exact, and cheaper than a float64 round trip.
func abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

// normOne evaluates the TM_CCOEFF_NORMED tail for one element from the
// three spilled lanes. cross and idiff are exact integers converted once
// to float32 (correctly rounded), so the variance term diff2 =
// idiff·invArea carries no cancellation at all — the exact-integer feed
// is what makes the float32 tail safe (and better conditioned than the
// former float64 replay of OpenCV's sequence). Every op is one correctly
// rounded float32 instruction on both ISAs; the float32 conversions pin
// the sequence (no contraction), and sqrt via float64 math.Sqrt rounds
// identically to the hardware float32 sqrt for float32 inputs.
func normOne(num, lane0, idiff, s2, numScale, varScale, eps, templNorm float32) float32 {
	num = float32(num - float32(lane0*numScale))
	diff2 := float32(idiff * varScale)
	lim := float32(eps * s2)
	if lim > 0.5 {
		lim = 0.5
	}
	var den float32
	if diff2 > lim {
		den = float32(float32(math.Sqrt(float64(diff2))) * templNorm)
	}
	switch {
	case abs32(num) < den:
		num /= den
	case abs32(num) < float32(den*1.125):
		if num > 0 {
			num = 1
		} else {
			num = -1
		}
	default:
		num = 0
	}
	return num
}

// normalizeBandGo processes result rows [y0, y1). Band-local column-sum
// rebuilds are bit-exact because all window statistics are exact
// integers; the spill folds the channels into three exact integer values
// per element — cross = Σ_k wndSum_k·tsum_k, idiff = area·wndSum2 −
// Σ_k wndSum_k² (the variance numerator, ≥ 0 per channel by
// Cauchy-Schwarz) and the raw wndSum2 — each converted once to float32
// (correctly rounded on every target), so chunking, banding and channel
// count never change a bit and the float32 tail sees no cancellation.
func normalizeBandGo(img []uint8, istride, iw int, cn, step, tw, th, rw, y0, y1 int,
	tsum *[4]int64, templNorm float32, corr []float32, result []float32) goExtrema {
	area := int64(tw) * int64(th)
	varScale := float32(1 / (float64(tw) * float64(th)))
	// cn=1 spills the raw window sum as lane0 (single exact conversion —
	// it stays below 2^52) and folds the template mean into the tail's
	// num-scale constant; cn>=3 spills the exact integer cross and scales
	// by 1/area. Either way lane0*numScale reproduces cross*invArea.
	numScale := varScale
	if cn == 1 {
		numScale = float32(float64(tsum[0]) / (float64(tw) * float64(th)))
	}
	cs := cn
	if step == 4 && (cn == 3 || cn == 4) {
		cs = 4
	}
	colSum := i32Pool.get(iw * cs)
	colSum2 := i64Pool.get(iw)
	defer i32Pool.put(colSum)
	defer i64Pool.put(colSum2)
	colBuildGo(colSum, colSum2, img, istride, iw, cn, cs, step, y0, th)

	wt := f32Pool.get(3 * normChunkGo)
	defer f32Pool.put(wt)

	ext := goExtrema{minV: math.MaxFloat32, maxV: -math.MaxFloat32, minY: y0, maxY: y0}
	const eps = float32(10.0 * 0x1p-23) // 10*FLT_EPSILON, exactly as OpenCV

	for y := y0; ; y++ {
		var s [4]int64
		var s2 int64
		for x := 0; x < tw; x++ {
			for k := 0; k < cn; k++ {
				s[k] += int64(colSum[x*cs+k])
			}
			s2 += colSum2[x]
		}
		rrow := result[y*rw : y*rw+rw]
		crow := corr[y*rw : y*rw+rw]

		// The row runs in chunks: each element spills its three exact
		// integer statistics as float32 lanes, and the chunk's tail math is
		// evaluated from the lanes, vectorized when the kernel is on.
		for x0 := 0; x0 < rw; x0 += normChunkGo {
			clen := min(normChunkGo, rw-x0)
			// Elements slide the window while x+1 < rw; only the row's very
			// last element doesn't, so the slide count is hoisted out of the
			// loop and the leftover element (at most one) spills afterwards.
			ns := min(clen, rw-1-x0)
			switch cn {
			case 1:
				lo, hi := colSum[x0:], colSum[x0+tw:]
				lo2, hi2 := colSum2[x0:], colSum2[x0+tw:]
				i := 0
				// The kernel's 32-bit product decomposition needs the
				// row-delta bound |hi2-lo2| < 2^31 (th ≤ 32767) AND window
				// sums below 2^31 (255·area < 2^31, i.e. area ≤ 8421504 —
				// the cn=1 stats cap is looser at 11.9M). Shapes beyond
				// either bound spill scalar, exactly.
				if vns := ns &^ 3; simd.Enabled && th <= 32767 && area <= 8_421_504 && vns > 0 {
					s[0], s2 = simd.SpillStats1(wt[:vns], normChunkGo,
						lo, hi, lo2, hi2, s[0], s2, area)
					i = vns
				}
				s0, t2 := s[0], s2
				for ; i < ns; i++ {
					wt[i] = float32(float64(s0))
					wt[normChunkGo+i] = float32(float64(area*t2 - s0*s0))
					wt[2*normChunkGo+i] = float32(float64(t2))
					s0 += int64(hi[i] - lo[i])
					t2 += hi2[i] - lo2[i]
				}
				s[0], s2 = s0, t2
			case 3:
				lo, hi := colSum[x0*cs:], colSum[(x0+tw)*cs:]
				lo2, hi2 := colSum2[x0:], colSum2[x0+tw:]
				i := 0
				// The kernel's area*d2 product needs |d2| = |Δ colSum2|
				// < 2^31, i.e. cn·255²·th < 2^31 → th ≤ 11008 for cn=3.
				if vns := ns &^ 3; simd.Enabled && cs == 4 && th <= 11008 && vns > 0 {
					s2 = simd.SpillStats4(wt[:vns], normChunkGo,
						lo, hi, lo2, hi2, &s, tsum, s2, area, false)
					i = vns
				}
				s0, s1, c2, t2 := s[0], s[1], s[2], s2
				t0, t1, tq := tsum[0], tsum[1], tsum[2]
				for ; i < ns; i++ {
					wt[i] = float32(float64(s0*t0 + s1*t1 + c2*tq))
					wt[normChunkGo+i] = float32(float64(area*t2 - s0*s0 - s1*s1 - c2*c2))
					wt[2*normChunkGo+i] = float32(float64(t2))
					j := i * cs
					s0 += int64(hi[j] - lo[j])
					s1 += int64(hi[j+1] - lo[j+1])
					c2 += int64(hi[j+2] - lo[j+2])
					t2 += hi2[i] - lo2[i]
				}
				s[0], s[1], s[2], s2 = s0, s1, c2, t2
			case 4:
				lo, hi := colSum[x0*cs:], colSum[(x0+tw)*cs:]
				lo2, hi2 := colSum2[x0:], colSum2[x0+tw:]
				i := 0
				// cn=4: 4·255²·th < 2^31 → th ≤ 8256.
				if vns := ns &^ 3; simd.Enabled && th <= 8256 && vns > 0 {
					s2 = simd.SpillStats4(wt[:vns], normChunkGo,
						lo, hi, lo2, hi2, &s, tsum, s2, area, true)
					i = vns
				}
				s0, s1, c2, s3, t2 := s[0], s[1], s[2], s[3], s2
				t0, t1, tq, t3 := tsum[0], tsum[1], tsum[2], tsum[3]
				for ; i < ns; i++ {
					wt[i] = float32(float64(s0*t0 + s1*t1 + c2*tq + s3*t3))
					wt[normChunkGo+i] = float32(float64(area*t2 - s0*s0 - s1*s1 - c2*c2 - s3*s3))
					wt[2*normChunkGo+i] = float32(float64(t2))
					j := i * 4
					s0 += int64(hi[j] - lo[j])
					s1 += int64(hi[j+1] - lo[j+1])
					c2 += int64(hi[j+2] - lo[j+2])
					s3 += int64(hi[j+3] - lo[j+3])
					t2 += hi2[i] - lo2[i]
				}
				s[0], s[1], s[2], s[3], s2 = s0, s1, c2, s3, t2
			default:
				for i := 0; i < ns; i++ {
					var cross, sq int64
					for k := 0; k < cn; k++ {
						cross += s[k] * tsum[k]
						sq += s[k] * s[k]
					}
					wt[i] = float32(float64(cross))
					wt[normChunkGo+i] = float32(float64(area*s2 - sq))
					wt[2*normChunkGo+i] = float32(float64(s2))
					x := x0 + i
					for k := 0; k < cn; k++ {
						s[k] += int64(colSum[(x+tw)*cs+k] - colSum[x*cs+k])
					}
					s2 += colSum2[x+tw] - colSum2[x]
				}
			}
			for i := ns; i < clen; i++ { // final result column: no slide
				var lane0, sq int64
				for k := 0; k < cn; k++ {
					lane0 += s[k] * tsum[k]
					sq += s[k] * s[k]
				}
				if cn == 1 {
					lane0 = s[0] // lane0 convention: raw window sum
				}
				wt[i] = float32(float64(lane0))
				wt[normChunkGo+i] = float32(float64(area*s2 - sq))
				wt[2*normChunkGo+i] = float32(float64(s2))
			}
			vlen := 0
			if simd.Enabled {
				vlen = clen &^ 7
				if vlen > 0 {
					simd.NormRow(rrow[x0:x0+vlen], crow[x0:x0+vlen], &wt[0],
						normChunkGo, vlen, numScale, varScale, eps, templNorm)
				}
			}
			for i := vlen; i < clen; i++ {
				rrow[x0+i] = normOne(crow[x0+i], wt[i], wt[normChunkGo+i],
					wt[2*normChunkGo+i], numScale, varScale, eps, templNorm)
			}
		}
		// Row min/max scan of the stored float32 values: ascending x with
		// strict compares keeps OpenCV's first-occurrence tie semantics.
		// The kernel returns the row prefix's first-occurrence extrema, so
		// merging with the running extrema by strict compare is identical
		// to scanning element-wise.
		xs := 0
		if simd.Enabled && rw >= 8 {
			xs = rw &^ 7
			mnV, mxV, mnI, mxI := simd.MinMaxRow(rrow[:xs])
			if mnV < ext.minV {
				ext.minV, ext.minX, ext.minY = mnV, mnI, y
			}
			if mxV > ext.maxV {
				ext.maxV, ext.maxX, ext.maxY = mxV, mxI, y
			}
		}
		for x := xs; x < rw; x++ {
			v := rrow[x]
			if v < ext.minV {
				ext.minV, ext.minX, ext.minY = v, x, y
			}
			if v > ext.maxV {
				ext.maxV, ext.maxX, ext.maxY = v, x, y
			}
		}
		if y+1 >= y1 {
			break
		}
		colSlideGo(colSum, colSum2, img[y*istride:], img[(y+th)*istride:], iw, cn, cs, step)
	}
	return ext
}

// templStats computes the per-channel template sums and the variance
// numerator varSum = Σ_k (area·Σt² − (Σt)²) — all exact integers (the
// matchU8 area bound keeps every product below 2^63), so the flat test is
// an exact ==0 and no float rounding enters until the single
// sqrt(varSum/area) that produces templNorm. It reads only the template,
// so matchU8 runs it before the correlation to short-circuit flat
// templates.
func templStats(tpl []uint8, tstride, tw, th, cn, step int) (tsum [4]int64, varSum int64) {
	area := int64(tw) * int64(th)
	for k := 0; k < cn; k++ {
		var s, s2 int64 // template statistics are exact integers
		for y := 0; y < th; y++ {
			row := tpl[y*tstride+k:]
			for x := 0; x < tw; x++ {
				v := int64(row[x*step])
				s += v
				s2 += v * v
			}
		}
		tsum[k] = s
		varSum += area*s2 - s*s
	}
	return tsum, varSum
}

func normalizeParallelGo(img []uint8, istride, iw, tw, th, cn, step, rw, rh, threads int,
	tsum *[4]int64, varSum int64,
	corr []float32, result []float32) (float32, int, int, float32, int, int) {
	// templNorm = sqrt(Σ_k templVar_k · area) = sqrt(varSum/area): one f64
	// sqrt of an exact integer ratio, converted once — a fixed sequence on
	// every target.
	templNorm := float32(math.Sqrt(float64(varSum) / (float64(tw) * float64(th))))

	nb := threads
	if maxb := rh / max(th, 32); maxb >= 1 && nb > maxb {
		nb = maxb
	}
	nb = max(min(nb, rh), 1)

	bandY := make([]int, nb+1)
	for b := 0; b <= nb; b++ {
		bandY[b] = rh * b / nb
	}
	ext := make([]goExtrema, nb)
	runParallel(nb, func(w int) {
		ext[w] = normalizeBandGo(img, istride, iw, cn, step, tw, th, rw,
			bandY[w], bandY[w+1], tsum, templNorm, corr, result)
	})
	r := ext[0]
	for b := 1; b < nb; b++ { // strict compares keep first occurrence
		if ext[b].minV < r.minV {
			r.minV, r.minX, r.minY = ext[b].minV, ext[b].minX, ext[b].minY
		}
		if ext[b].maxV > r.maxV {
			r.maxV, r.maxX, r.maxY = ext[b].maxV, ext[b].maxX, ext[b].maxY
		}
	}
	return r.minV, r.minX, r.minY, r.maxV, r.maxX, r.maxY
}

// ----------------------------------------------------------- entrypoint --

// statsCap returns the largest template area whose exact-integer window
// statistics fit int64: cn·65025·area² < 2^63, i.e. ⌊√(2^63/65025/cn)⌋ —
// 11.9M pixels single-channel down to 5.95M at cn=4 (a ~2400x2400 RGBA
// template). Far beyond any real workload, but asserted, not assumed.
func statsCap(cn int) int64 {
	return [5]int64{0, 11_909_805, 8_421_504, 6_876_129, 5_954_902}[cn]
}

func matchU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if cn < 1 || cn > 4 || step < cn || tw < 1 || th < 1 || tw > iw || th > ih ||
		istride < iw*step || tstride < tw*step {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d, cn=%d step=%d)", tw, th, iw, ih, cn, step))
	}
	if int64(tw)*int64(th) > statsCap(cn) {
		panic(fmt.Sprintf("cvmatch: template area %dx%d exceeds the exact-statistics bound for cn=%d", tw, th, cn))
	}
	threads = clampThreads(threads)
	rw, rh := iw-tw+1, ih-th+1
	tsum, varSum := templStats(tpl, tstride, tw, th, cn, step)
	if varSum == 0 { // exactly flat template: scores 1 everywhere
		if result != nil {
			for i := range result[:rw*rh] {
				result[i] = 1
			}
		}
		return 1, 0, 0, 1, 0, 0
	}
	res := result
	if res == nil {
		res = f32Pool.get(rw * rh)
		defer f32Pool.put(res)
	}
	p := newGoPlan(tw, th, rw, rh)
	crossCorrGo(img, istride, tpl, tstride, step, cn, tw, th, rw, rh, threads, p, res)
	return normalizeParallelGo(img, istride, iw, tw, th, cn, step, rw, rh, threads, &tsum, varSum, res, res)
}
