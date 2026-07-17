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
	"sync"

	"github.com/hkloudou/cvmatch/internal/simd"
)

const maxThreads = 16

// Scratch pools: every pooled buffer is fully overwritten before use
// (block spectra include their zero padding, column sums are cleared in
// colBuildGo, the internal result map is stored before it is read), so
// dirty reuse is safe and steady-state matching allocates nothing.
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

// makeTwiddles fills tw[half+j] = exp(-pi*i*j/half) for each power-of-two
// stage.
func makeTwiddles(n int) []complex64 {
	tw := make([]complex64, n)
	tw[0] = 1
	for half := 1; half < n; half <<= 1 {
		for j := 0; j < half; j++ {
			c, s := sincospiFrac(j, half)
			tw[half+j] = complex(c, -s)
		}
	}
	return tw
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
	tw    []complex64
	pairs []int32
}

func fftTables(n int) *fftTab {
	if n > 4096 {
		return &fftTab{makeTwiddles(n), makeSwapPairs(n)}
	}
	fftTabMu.Lock()
	t := fftTabs[n]
	if t == nil {
		t = &fftTab{makeTwiddles(n), makeSwapPairs(n)}
		fftTabs[n] = t
	}
	fftTabMu.Unlock()
	return t
}

// bfly applies one radix-2 butterfly with explicit float32 single-rounding
// semantics: the products are rounded to float32 before the add/sub (the
// gc compiler would otherwise evaluate complex64 products through float64
// intermediates on some architectures, and may fuse mul-adds on others;
// the float32 conversions pin both down). The SIMD kernels implement
// precisely these ops.
func bfly(p, q *complex64, wr, wi float32) {
	qv := *q
	vr := float32(real(qv)*wr) - float32(imag(qv)*wi)
	vi := float32(real(qv)*wi) + float32(imag(qv)*wr)
	u := *p
	*p = complex(real(u)+vr, imag(u)+vi)
	*q = complex(real(u)-vr, imag(u)-vi)
}

// bflyV is bfly by value — identical op sequence, but operands stay in
// registers so fused stage pairs can chain butterflies without a memory
// round trip between layers.
func bflyV(u, q complex64, wr, wi float32) (complex64, complex64) {
	vr := float32(real(q)*wr) - float32(imag(q)*wi)
	vi := float32(real(q)*wi) + float32(imag(q)*wr)
	return complex(real(u)+vr, imag(u)+vi), complex(real(u)-vr, imag(u)-vi)
}

// twdir returns w's components with the inverse-direction sign applied to
// the imaginary part. Multiplying by ±1 is an exact sign operation, so
// this replaces the former per-butterfly s*imag multiply bit-identically.
func twdir(w complex64, inverse bool) (float32, float32) {
	if inverse {
		return real(w), -imag(w)
	}
	return real(w), imag(w)
}

func fftGo(a []complex64, tw []complex64, pairs []int32, inverse bool) {
	n := len(a)
	for k := 0; k+1 < len(pairs); k += 2 {
		i, j := pairs[k], pairs[k+1]
		a[i], a[j] = a[j], a[i]
	}
	if n >= 8 && simd.Enabled {
		simd.FFTStages(a, tw, inverse)
		return
	}
	// The half=1,2 stages are fused into one quad pass (both layers of a
	// closed 4-element group chain in registers), and the remaining
	// cascade fuses stage pairs the same way — exactly the structure of
	// the asm kernels. Each element still sees exactly the arithmetic of
	// the generic stage loop, so results are bit-identical.
	if n >= 4 {
		w1r, w1i := twdir(tw[1], inverse)
		w2ar, w2ai := twdir(tw[2], inverse)
		w2br, w2bi := twdir(tw[3], inverse)
		for i := 0; i < n; i += 4 {
			p := a[i : i+4 : i+4]
			x0, x1 := bflyV(p[0], p[1], w1r, w1i)
			x2, x3 := bflyV(p[2], p[3], w1r, w1i)
			x0, x2 = bflyV(x0, x2, w2ar, w2ai)
			x1, x3 = bflyV(x1, x3, w2br, w2bi)
			p[0], p[1], p[2], p[3] = x0, x1, x2, x3
		}
	} else if n == 2 {
		w0r, w0i := twdir(tw[1], inverse)
		bfly(&a[0], &a[1], w0r, w0i)
	}
	half := 4
	for ; half*2 < n; half *= 4 {
		for i := 0; i < n; i += half * 4 {
			for j := 0; j < half; j++ {
				w1r, w1i := twdir(tw[half+j], inverse)
				w2ar, w2ai := twdir(tw[2*half+j], inverse)
				w2br, w2bi := twdir(tw[3*half+j], inverse)
				p := a[i+j : i+j+3*half+1 : i+j+3*half+1]
				x0, x1 := bflyV(p[0], p[half], w1r, w1i)
				x2, x3 := bflyV(p[2*half], p[3*half], w1r, w1i)
				x0, x2 = bflyV(x0, x2, w2ar, w2ai)
				x1, x3 = bflyV(x1, x3, w2br, w2bi)
				p[0], p[half], p[2*half], p[3*half] = x0, x1, x2, x3
			}
		}
	}
	for ; half < n; half <<= 1 {
		w := tw[half : half*2]
		for i := 0; i < n; i += half << 1 {
			p := a[i : i+half : i+half]
			q := a[i+half : i+half*2 : i+half*2]
			if inverse {
				for j := 0; j < half; j++ {
					bfly(&p[j], &q[j], real(w[j]), -imag(w[j]))
				}
			} else {
				for j := 0; j < half; j++ {
					bfly(&p[j], &q[j], real(w[j]), imag(w[j]))
				}
			}
		}
	}
}

// fftColsGo transforms the columns of a row-major [n x width] array; the
// butterfly inner loop runs across contiguous row elements.
func fftColsGo(d []complex64, n, width int, tw []complex64, pairs []int32, inverse bool, tmp []complex64) {
	for k := 0; k+1 < len(pairs); k += 2 {
		i, j := int(pairs[k]), int(pairs[k+1])
		ri := d[i*width : i*width+width]
		rj := d[j*width : j*width+width]
		copy(tmp[:width], ri)
		copy(ri, rj)
		copy(rj, tmp[:width])
	}
	// Fused stage pairs: rows {i+j, +half, +2half, +3half} are closed
	// under stages half and 2*half, so each pass streams the matrix once
	// instead of twice. Butterfly order within a quad follows the stage
	// order, so results are bit-identical — in the kernels and in the
	// register-chained scalar loop alike.
	half := 1
	for ; half*2 < n; half *= 4 {
		for i := 0; i < n; i += half * 4 {
			for j := 0; j < half; j++ {
				w1r, w1i := twdir(tw[half+j], inverse)
				w2ar, w2ai := twdir(tw[2*half+j], inverse)
				w2br, w2bi := twdir(tw[3*half+j], inverse)
				r := i + j
				p0 := d[r*width : r*width+width]
				p1 := d[(r+half)*width : (r+half)*width+width]
				p2 := d[(r+half*2)*width : (r+half*2)*width+width]
				p3 := d[(r+half*3)*width : (r+half*3)*width+width]
				if simd.Enabled {
					simd.FFTCols4(p0, p1, p2, p3,
						complex(w1r, w1i), complex(w2ar, w2ai), complex(w2br, w2bi))
					continue
				}
				for c := range p0 {
					x0, x1 := bflyV(p0[c], p1[c], w1r, w1i)
					x2, x3 := bflyV(p2[c], p3[c], w1r, w1i)
					x0, x2 = bflyV(x0, x2, w2ar, w2ai)
					x1, x3 = bflyV(x1, x3, w2br, w2bi)
					p0[c], p1[c], p2[c], p3[c] = x0, x1, x2, x3
				}
			}
		}
	}
	for ; half < n; half <<= 1 {
		for i := 0; i < n; i += half << 1 {
			for j := 0; j < half; j++ {
				wr, wi := twdir(tw[half+j], inverse)
				p := d[(i+j)*width : (i+j)*width+width]
				q := d[(i+j+half)*width : (i+j+half)*width+width]
				if simd.Enabled {
					simd.FFTColsBfly(p, q, complex(wr, wi))
					continue
				}
				for c := range p {
					bfly(&p[c], &q[c], wr, wi)
				}
			}
		}
	}
}

// -------------------------------------------------------- FFT plan/blocks --

type goPlan struct {
	dftW, dftH, hw int
	blockW, blockH int
	twW, twH       []complex64
	prW, prH       []int32
}

// newGoPlan mirrors plan_init: OpenCV's crossCorr block sizing with
// power-of-two DFT dims and the same scratch-memory area cap.
func newGoPlan(tw, th, rw, rh int) *goPlan {
	bw := int(float64(tw)*4.5 + 0.5)
	bh := int(float64(th)*4.5 + 0.5)
	if bw < 256-tw+1 {
		bw = 256 - tw + 1
	}
	if bh < 256-th+1 {
		bh = 256 - th + 1
	}
	if bw > rw {
		bw = rw
	}
	if bh > rh {
		bh = rh
	}
	dftW, dftH := nextPow2(bw+tw-1), nextPow2(bh+th-1)
	if dftW < 2 {
		dftW = 2
	}
	if dftH < 2 {
		dftH = 2
	}
	for int64(dftW)*int64(dftH) > 1<<21 {
		if dftW >= dftH && dftW>>1 >= tw+1 {
			dftW >>= 1
		} else if dftH>>1 >= th+1 {
			dftH >>= 1
		} else {
			break
		}
	}
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
	p.twW, p.prW = tabW.tw, tabW.pairs
	if dftH == dftW {
		p.twH, p.prH = p.twW, p.prW
	} else {
		tabH := fftTables(dftH)
		p.twH, p.prH = tabH.tw, tabH.pairs
	}
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
	fftGo(z, p.twW, p.prW, false)
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
	fftColsGo(spec, p.dftH, hw, p.twH, p.prH, false, z)
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
	fftGo(z, p.twW, p.prW, true)
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
	fftColsGo(spec, p.dftH, p.hw, p.twH, p.prH, true, z)
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
	tspec := cplxPool.get(cn * specN)
	z0 := cplxPool.get(p.dftW)
	scale := 1 / (float32(p.dftW) * float32(p.dftH))
	for k := 0; k < cn; k++ {
		ts := tspec[k*specN : (k+1)*specN]
		blockForwardGo(tpl[k:], tstride, step, 0, 0, tw, th, p, ts, z0, 1)
		for i := range ts {
			ts[i] = complex(real(ts[i])*scale, imag(ts[i])*scale)
		}
	}
	cplxPool.put(z0)
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
	team := threads / nw
	runParallel(nw, func(w int) {
		spec := cplxPool.get(specN)
		z := cplxPool.get(p.dftW)
		for t := w; t < ntiles; t += nw {
			x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
			bw, bh := min(p.blockW, rw-x0), min(p.blockH, rh-y0)
			for k := 0; k < cn; k++ {
				blockForwardGo(img[k:], istride, step, x0, y0, bw+tw-1, bh+th-1, p, spec, z, team)
				ts := tspec[k*specN : (k+1)*specN]
				if team <= 1 {
					mulConjGo(spec, ts)
				} else {
					runParallel(team, func(u int) {
						lo := u * specN / team
						hi := (u + 1) * specN / team
						mulConjGo(spec[lo:hi], ts[lo:hi])
					})
				}
				blockInverseEmitGo(p, spec, z, result, rw, x0, y0, bw, bh, k > 0, team)
			}
		}
		cplxPool.put(spec)
		cplxPool.put(z)
	})
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

// normOne evaluates the TM_CCOEFF_NORMED tail for one element. The float64
// conversions around products pin each op to one rounding (Go may
// otherwise fuse mul-adds on some architectures).
func normOne(num, wndMean2, s2d, invArea, eps, templNorm float64) float32 {
	wndMean2 = float64(wndMean2 * invArea)
	diff2 := s2d - wndMean2
	if diff2 < 0 {
		diff2 = 0
	}
	lim := eps * s2d
	if lim > 0.5 {
		lim = 0.5
	}
	den := 0.0
	if diff2 > lim {
		den = math.Sqrt(diff2) * templNorm
	}
	switch {
	case math.Abs(num) < den:
		num /= den
	case math.Abs(num) < den*1.125:
		if num > 0 {
			num = 1
		} else {
			num = -1
		}
	default:
		num = 0
	}
	return float32(num)
}

// normalizeBandGo processes result rows [y0, y1). Band-local column-sum
// rebuilds are bit-exact because all window statistics are exact integers.
func normalizeBandGo(img []uint8, istride, iw int, cn, step, tw, th, rw, y0, y1 int,
	mean *[4]float64, templNorm float64, corr []float32, result []float32) goExtrema {
	invArea := 1 / (float64(tw) * float64(th))
	cs := cn
	if step == 4 && (cn == 3 || cn == 4) {
		cs = 4
	}
	colSum := i32Pool.get(iw * cs)
	colSum2 := i64Pool.get(iw)
	defer i32Pool.put(colSum)
	defer i64Pool.put(colSum2)
	colBuildGo(colSum, colSum2, img, istride, iw, cn, cs, step, y0, th)

	wt := f64Pool.get((cn + 1) * normChunkGo)
	defer f64Pool.put(wt)
	q2 := wt[cn*normChunkGo:]
	useKernel := simd.Enabled && cn != 2

	ext := goExtrema{minV: math.MaxFloat32, maxV: -math.MaxFloat32, minY: y0, maxY: y0}
	const eps = 10.0 * 0x1p-23 // 10*FLT_EPSILON, exactly as OpenCV

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

		// The row runs in chunks: the (exact integer) window sums are spilled
		// as float64 lanes — a lossless conversion, so chunking never changes
		// a value — and each chunk's tail math is evaluated from the lanes,
		// vectorized when the kernel covers this channel count.
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
				if useKernel && ns > 0 {
					s[0], s2 = simd.SlideSpill1(wt[:ns], q2[:ns], lo, hi, lo2, hi2, s[0], s2)
				} else {
					s0, t2 := s[0], s2
					for i := 0; i < ns; i++ {
						wt[i] = float64(s0)
						q2[i] = float64(t2)
						s0 += int64(hi[i] - lo[i])
						t2 += hi2[i] - lo2[i]
					}
					s[0], s2 = s0, t2
				}
			case 3:
				lo, hi := colSum[x0*cs:], colSum[(x0+tw)*cs:]
				lo2, hi2 := colSum2[x0:], colSum2[x0+tw:]
				s0, s1, sq, t2 := s[0], s[1], s[2], s2
				for i := 0; i < ns; i++ {
					wt[i] = float64(s0)
					wt[normChunkGo+i] = float64(s1)
					wt[2*normChunkGo+i] = float64(sq)
					q2[i] = float64(t2)
					j := i * cs
					s0 += int64(hi[j] - lo[j])
					s1 += int64(hi[j+1] - lo[j+1])
					sq += int64(hi[j+2] - lo[j+2])
					t2 += hi2[i] - lo2[i]
				}
				s[0], s[1], s[2], s2 = s0, s1, sq, t2
			case 4:
				lo, hi := colSum[x0*cs:], colSum[(x0+tw)*cs:]
				lo2, hi2 := colSum2[x0:], colSum2[x0+tw:]
				s0, s1, sq, s3, t2 := s[0], s[1], s[2], s[3], s2
				for i := 0; i < ns; i++ {
					wt[i] = float64(s0)
					wt[normChunkGo+i] = float64(s1)
					wt[2*normChunkGo+i] = float64(sq)
					wt[3*normChunkGo+i] = float64(s3)
					q2[i] = float64(t2)
					j := i * 4
					s0 += int64(hi[j] - lo[j])
					s1 += int64(hi[j+1] - lo[j+1])
					sq += int64(hi[j+2] - lo[j+2])
					s3 += int64(hi[j+3] - lo[j+3])
					t2 += hi2[i] - lo2[i]
				}
				s[0], s[1], s[2], s[3], s2 = s0, s1, sq, s3, t2
			default:
				for i := 0; i < ns; i++ {
					for k := 0; k < cn; k++ {
						wt[k*normChunkGo+i] = float64(s[k])
					}
					q2[i] = float64(s2)
					x := x0 + i
					for k := 0; k < cn; k++ {
						s[k] += int64(colSum[(x+tw)*cs+k] - colSum[x*cs+k])
					}
					s2 += colSum2[x+tw] - colSum2[x]
				}
			}
			for i := ns; i < clen; i++ { // final result column: no slide
				for k := 0; k < cn; k++ {
					wt[k*normChunkGo+i] = float64(s[k])
				}
				q2[i] = float64(s2)
			}
			vlen := 0
			if useKernel {
				vlen = clen &^ 3
				if vlen > 0 {
					simd.NormRow(rrow[x0:x0+vlen], crow[x0:x0+vlen], &wt[0],
						normChunkGo, vlen, cn, mean, invArea, eps, templNorm)
				}
			}
			for i := vlen; i < clen; i++ {
				num := float64(crow[x0+i])
				wndMean2 := 0.0
				for k := 0; k < cn; k++ {
					t := wt[k*normChunkGo+i]
					wndMean2 += float64(t * t)
					num -= float64(t * mean[k])
				}
				rrow[x0+i] = normOne(num, wndMean2, q2[i], invArea, eps, templNorm)
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

func normalizeParallelGo(img []uint8, istride, iw int, tpl []uint8, tstride, tw, th, cn, step, rw, rh, threads int,
	corr []float32, result []float32) (float32, int, int, float32, int, int) {
	invArea := 1 / (float64(tw) * float64(th))
	var mean [4]float64
	templNorm := 0.0
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
		mean[k] = float64(s) * invArea
		templNorm += float64(float64(s2)*invArea) - float64(mean[k]*mean[k])
	}
	if templNorm < 0x1p-52 { // DBL_EPSILON: flat template
		for i := range result[:rw*rh] {
			result[i] = 1
		}
		return 1, 0, 0, 1, 0, 0
	}
	templNorm = math.Sqrt(templNorm * (float64(tw) * float64(th)))

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
			bandY[w], bandY[w+1], &mean, templNorm, corr, result)
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

func matchU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if cn < 1 || cn > 4 || step < cn || tw < 1 || th < 1 || tw > iw || th > ih ||
		istride < iw*step || tstride < tw*step {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d, cn=%d step=%d)", tw, th, iw, ih, cn, step))
	}
	threads = clampThreads(threads)
	rw, rh := iw-tw+1, ih-th+1
	res := result
	if res == nil {
		res = f32Pool.get(rw * rh)
		defer f32Pool.put(res)
	}
	p := newGoPlan(tw, th, rw, rh)
	crossCorrGo(img, istride, tpl, tstride, step, cn, tw, th, rw, rh, threads, p, res)
	return normalizeParallelGo(img, istride, iw, tpl, tstride, tw, th, cn, step, rw, rh, threads, res, res)
}
