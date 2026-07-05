package cvmatch

// Pure-Go port of cvmatch.c, selected automatically when cgo is off (see
// impl_nocgo.go). It runs the same algorithm — block-FFT cross-correlation
// with OpenCV's tile heuristic, sliding-column-sum normalization with
// OpenCV's exact guards, fused minMaxLoc, tile/band parallelism with
// bit-identical output for any worker count — and is always compiled so
// tests can compare it against the C core when cgo is on.

import (
	"fmt"
	"math"
	"math/bits"
	"sync"
)

const maxThreadsGo = 16

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
	if n > maxThreadsGo {
		return maxThreadsGo
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

// makeTwiddles fills tw[half+j] = exp(-pi*i*j/half) for each power-of-two
// stage, matching the C layout.
func makeTwiddles(n int) []complex64 {
	tw := make([]complex64, n)
	tw[0] = 1
	for half := 1; half < n; half <<= 1 {
		step := -math.Pi / float64(half)
		for j := 0; j < half; j++ {
			s, c := math.Sincos(step * float64(j))
			tw[half+j] = complex(float32(c), float32(s))
		}
	}
	return tw
}

func makeBitrev(n int) []int {
	br := make([]int, n)
	for i := 1; i < n; i++ {
		br[i] = br[i>>1] >> 1
		if i&1 != 0 {
			br[i] |= n >> 1
		}
	}
	return br
}

func fftGo(a []complex64, tw []complex64, br []int, inverse bool) {
	n := len(a)
	for i, j := range br {
		if j > i {
			a[i], a[j] = a[j], a[i]
		}
	}
	for half := 1; half < n; half <<= 1 {
		w := tw[half : half*2]
		for i := 0; i < n; i += half << 1 {
			p := a[i : i+half : i+half]
			q := a[i+half : i+half*2 : i+half*2]
			if inverse {
				for j := 0; j < half; j++ {
					wj := complex(real(w[j]), -imag(w[j]))
					v := q[j] * wj
					u := p[j]
					p[j] = u + v
					q[j] = u - v
				}
			} else {
				for j := 0; j < half; j++ {
					v := q[j] * w[j]
					u := p[j]
					p[j] = u + v
					q[j] = u - v
				}
			}
		}
	}
}

// fftColsGo transforms the columns of a row-major [n x width] array; the
// butterfly inner loop runs across contiguous row elements.
func fftColsGo(d []complex64, n, width int, tw []complex64, br []int, inverse bool, tmp []complex64) {
	for i, j := range br {
		if j > i {
			ri := d[i*width : i*width+width]
			rj := d[j*width : j*width+width]
			copy(tmp[:width], ri)
			copy(ri, rj)
			copy(rj, tmp[:width])
		}
	}
	for half := 1; half < n; half <<= 1 {
		for i := 0; i < n; i += half << 1 {
			for j := 0; j < half; j++ {
				w := tw[half+j]
				if inverse {
					w = complex(real(w), -imag(w))
				}
				p := d[(i+j)*width : (i+j)*width+width]
				q := d[(i+j+half)*width : (i+j+half)*width+width]
				for c := range p {
					v := q[c] * w
					u := p[c]
					p[c] = u + v
					q[c] = u - v
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
	brW, brH       []int
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
	p.twW, p.brW = makeTwiddles(dftW), makeBitrev(dftW)
	if dftH == dftW {
		p.twH, p.brH = p.twW, p.brW
	} else {
		p.twH, p.brH = makeTwiddles(dftH), makeBitrev(dftH)
	}
	return p
}

// blockForwardGo: real 2D forward DFT of one channel of a uint8 block, two
// real rows packed per complex FFT.
func blockForwardGo(chanBase []uint8, stride, step, x0, y0, loadW, loadH int, p *goPlan, spec, z []complex64) {
	n, hw, mask := p.dftW, p.hw, p.dftW-1
	for r := 0; r < p.dftH; r += 2 {
		sa := spec[r*hw : r*hw+hw]
		sb := spec[(r+1)*hw : (r+1)*hw+hw]
		if r >= loadH {
			tail := spec[r*hw : p.dftH*hw]
			for i := range tail {
				tail[i] = 0
			}
			break
		}
		ra := chanBase[(y0+r)*stride+x0*step:]
		if r+1 < loadH {
			rb := chanBase[(y0+r+1)*stride+x0*step:]
			for x := 0; x < loadW; x++ {
				z[x] = complex(float32(ra[x*step]), float32(rb[x*step]))
			}
		} else {
			for x := 0; x < loadW; x++ {
				z[x] = complex(float32(ra[x*step]), 0)
			}
		}
		for x := loadW; x < n; x++ {
			z[x] = 0
		}
		fftGo(z, p.twW, p.brW, false)
		for k := 0; k < hw; k++ {
			zk, zn := z[k], z[(n-k)&mask]
			sa[k] = complex(0.5*(real(zk)+real(zn)), 0.5*(imag(zk)-imag(zn)))
			sb[k] = complex(0.5*(imag(zk)+imag(zn)), 0.5*(real(zn)-real(zk)))
		}
	}
	fftColsGo(spec, p.dftH, hw, p.twH, p.brH, false, z)
}

func mulConjGo(spec, tspec []complex64) {
	for i, a := range spec {
		b := tspec[i]
		spec[i] = complex(real(a)*real(b)+imag(a)*imag(b), imag(a)*real(b)-real(a)*imag(b))
	}
}

func blockInverseEmitGo(p *goPlan, spec, z []complex64, res []float32, rw, x0, y0, bw, bh int, add bool) {
	fftColsGo(spec, p.dftH, p.hw, p.twH, p.brH, true, z)
	n, hw := p.dftW, p.hw
	for r := 0; r < bh; r += 2 {
		sa := spec[r*hw : r*hw+hw]
		sb := spec[(r+1)*hw : (r+1)*hw+hw]
		for k := 0; k < hw; k++ {
			z[k] = complex(real(sa[k])-imag(sb[k]), imag(sa[k])+real(sb[k]))
		}
		for k := hw; k < n; k++ {
			m := n - k
			z[k] = complex(real(sa[m])+imag(sb[m]), real(sb[m])-imag(sa[m]))
		}
		fftGo(z, p.twW, p.brW, true)
		o := res[(y0+r)*rw+x0:]
		if add {
			for x := 0; x < bw; x++ {
				o[x] += real(z[x])
			}
			if r+1 < bh {
				o2 := res[(y0+r+1)*rw+x0:]
				for x := 0; x < bw; x++ {
					o2[x] += imag(z[x])
				}
			}
		} else {
			for x := 0; x < bw; x++ {
				o[x] = real(z[x])
			}
			if r+1 < bh {
				o2 := res[(y0+r+1)*rw+x0:]
				for x := 0; x < bw; x++ {
					o2[x] = imag(z[x])
				}
			}
		}
	}
}

// crossCorrGo runs the tile-parallel raw cross-correlation. Each tile owns a
// disjoint result region and runs every channel in order, so per-element
// arithmetic is identical for any worker count.
func crossCorrGo(img []uint8, istride int, tpl []uint8, tstride, step, cn, tw, th, rw, rh, threads int, p *goPlan, result []float32) {
	specN := p.dftH * p.hw
	tspec := make([]complex64, cn*specN)
	z0 := make([]complex64, p.dftW)
	scale := complex(1/(float32(p.dftW)*float32(p.dftH)), 0)
	for k := 0; k < cn; k++ {
		ts := tspec[k*specN : (k+1)*specN]
		blockForwardGo(tpl[k:], tstride, step, 0, 0, tw, th, p, ts, z0)
		for i := range ts {
			ts[i] *= scale
		}
	}
	ntx := (rw + p.blockW - 1) / p.blockW
	nty := (rh + p.blockH - 1) / p.blockH
	ntiles := ntx * nty
	nw := threads
	if nw > ntiles {
		nw = ntiles
	}
	runParallel(nw, func(w int) {
		spec := make([]complex64, specN)
		z := make([]complex64, p.dftW)
		for t := w; t < ntiles; t += nw {
			x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
			bw, bh := min(p.blockW, rw-x0), min(p.blockH, rh-y0)
			for k := 0; k < cn; k++ {
				blockForwardGo(img[k:], istride, step, x0, y0, bw+tw-1, bh+th-1, p, spec, z)
				mulConjGo(spec, tspec[k*specN:(k+1)*specN])
				blockInverseEmitGo(p, spec, z, result, rw, x0, y0, bw, bh, k > 0)
			}
		}
	})
}

// ------------------------------------------------- normalization + scan --

type goExtrema struct {
	minV, maxV             float32
	minX, minY, maxX, maxY int
}

// normalizeBandGo processes result rows [y0, y1). Exactly one of corrF/corrD
// is non-nil. Band-local column-sum rebuilds are bit-exact because all sums
// are integer-valued doubles.
func normalizeBandGo(img []uint8, istride, iw int, cn, step, tw, th, rw, y0, y1 int,
	mean *[4]float64, templNorm float64, corrF []float32, corrD []float64, result []float32) goExtrema {
	invArea := 1 / (float64(tw) * float64(th))
	colSum := make([]float64, iw*cn)
	colSum2 := make([]float64, iw)
	for y := y0; y < y0+th; y++ {
		row := img[y*istride:]
		for x := 0; x < iw; x++ {
			for k := 0; k < cn; k++ {
				v := float64(row[x*step+k])
				colSum[x*cn+k] += v
				colSum2[x] += v * v
			}
		}
	}

	ext := goExtrema{minV: math.MaxFloat32, maxV: -math.MaxFloat32, minY: y0, maxY: y0}
	const eps = 10.0 * 1.1920929e-07 // 10*FLT_EPSILON

	for y := y0; ; y++ {
		var s [4]float64
		s2 := 0.0
		for x := 0; x < tw; x++ {
			for k := 0; k < cn; k++ {
				s[k] += colSum[x*cn+k]
			}
			s2 += colSum2[x]
		}
		rrow := result[y*rw : y*rw+rw]
		var cfr []float32
		var cdr []float64
		if corrF != nil {
			cfr = corrF[y*rw : y*rw+rw]
		} else {
			cdr = corrD[y*rw : y*rw+rw]
		}
		for x := 0; ; x++ {
			var num float64
			if cfr != nil {
				num = float64(cfr[x])
			} else {
				num = cdr[x]
			}
			wndMean2 := 0.0
			for k := 0; k < cn; k++ {
				t := s[k]
				wndMean2 += t * t
				num -= t * mean[k]
			}
			wndMean2 *= invArea
			diff2 := s2 - wndMean2
			if diff2 < 0 {
				diff2 = 0
			}
			lim := eps * s2
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
			// Compare the float32 value actually stored in the result map,
			// matching OpenCV minMaxLoc scanning the rounded CV_32F data.
			v := float32(num)
			rrow[x] = v
			if v < ext.minV {
				ext.minV, ext.minX, ext.minY = v, x, y
			}
			if v > ext.maxV {
				ext.maxV, ext.maxX, ext.maxY = v, x, y
			}
			if x+1 >= rw {
				break
			}
			for k := 0; k < cn; k++ {
				s[k] += colSum[(x+tw)*cn+k] - colSum[x*cn+k]
			}
			s2 += colSum2[x+tw] - colSum2[x]
		}
		if y+1 >= y1 {
			break
		}
		sub := img[y*istride:]
		add := img[(y+th)*istride:]
		for x := 0; x < iw; x++ {
			for k := 0; k < cn; k++ {
				a := float64(add[x*step+k])
				b := float64(sub[x*step+k])
				colSum[x*cn+k] += a - b
				colSum2[x] += a*a - b*b
			}
		}
	}
	return ext
}

func normalizeParallelGo(img []uint8, istride, iw int, tpl []uint8, tstride, tw, th, cn, step, rw, rh, threads int,
	corrF []float32, corrD []float64, result []float32) (float32, int, int, float32, int, int) {
	invArea := 1 / (float64(tw) * float64(th))
	var mean [4]float64
	templNorm := 0.0
	for k := 0; k < cn; k++ {
		s, s2 := 0.0, 0.0
		for y := 0; y < th; y++ {
			row := tpl[y*tstride+k:]
			for x := 0; x < tw; x++ {
				v := float64(row[x*step])
				s += v
				s2 += v * v
			}
		}
		mean[k] = s * invArea
		templNorm += s2*invArea - mean[k]*mean[k]
	}
	if templNorm < 2.220446049250313e-16 { // DBL_EPSILON: flat template
		for i := range result[:rw*rh] {
			result[i] = 1
		}
		return 1, 0, 0, 1, 0, 0
	}
	templNorm = math.Sqrt(templNorm * float64(tw) * float64(th))

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
			bandY[w], bandY[w+1], &mean, templNorm, corrF, corrD, result)
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

// -------------------------------------------- NTT (exact) correlation ----

// Montgomery arithmetic modulo p = 29*2^57+1, mirroring the C core exactly.

const nttP uint64 = 4179340454199820289

type mont struct {
	pinv, r2, one uint64
}

func newMont() mont {
	inv := nttP // Newton iteration for p^-1 mod 2^64
	for i := 0; i < 6; i++ {
		inv *= 2 - nttP*inv
	}
	var m mont
	m.pinv = -inv
	m.one = uint64(new128mod(1, 0)) // 2^64 mod p
	m.r2 = montModMul(m.one, m.one) // (2^64)^2 mod p, via plain 128-bit mod
	return m
}

// new128mod computes (hi*2^64 + lo) mod p using 128-bit division-free math.
func new128mod(hi, lo uint64) uint64 {
	// only used at init; do it the simple way
	var r uint64
	for i := 127; i >= 0; i-- {
		r <<= 1
		if r >= nttP {
			r -= nttP
		}
		var bit uint64
		if i >= 64 {
			bit = (hi >> (i - 64)) & 1
		} else {
			bit = (lo >> i) & 1
		}
		r |= 0 // keep shape
		if bit != 0 {
			r++
			if r >= nttP {
				r -= nttP
			}
		}
	}
	return r
}

// montModMul is a plain (a*b) mod p for init-time constants.
func montModMul(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return new128mod(hi, lo)
}

func (m *mont) mul(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	q := lo * m.pinv
	mh, ml := bits.Mul64(q, nttP)
	_, carry := bits.Add64(lo, ml, 0)
	r, _ := bits.Add64(hi, mh, carry)
	if r >= nttP {
		r -= nttP
	}
	return r
}

func nttAdd(a, b uint64) uint64 {
	r := a + b
	if r >= nttP {
		r -= nttP
	}
	return r
}

func nttSub(a, b uint64) uint64 {
	if a >= b {
		return a - b
	}
	return a + nttP - b
}

func (m *mont) pow(baseM, e uint64) uint64 {
	r, b := m.one, baseM
	for e != 0 {
		if e&1 != 0 {
			r = m.mul(r, b)
		}
		b = m.mul(b, b)
		e >>= 1
	}
	return r
}

func (m *mont) toMont(a uint64) uint64   { return m.mul(a, m.r2) }
func (m *mont) fromMont(a uint64) uint64 { return m.mul(a, 1) }

func (m *mont) generator() uint64 {
	for g := uint64(2); ; g++ {
		gm := m.toMont(g)
		if m.pow(gm, (nttP-1)/2) != m.one && m.pow(gm, (nttP-1)/29) != m.one {
			return gm
		}
	}
}

func makeNttTab(m *mont, genM uint64, n int, inverse bool) []uint64 {
	tab := make([]uint64, n)
	tab[0] = m.one
	for half := 1; half < n; half <<= 1 {
		e := (nttP - 1) / uint64(2*half)
		if inverse {
			e = nttP - 1 - e
		}
		w := m.pow(genM, e)
		cur := m.one
		for j := 0; j < half; j++ {
			tab[half+j] = cur
			cur = m.mul(cur, w)
		}
	}
	return tab
}

func nttGo(m *mont, a []uint64, tab []uint64, br []int) {
	n := len(a)
	for i, j := range br {
		if j > i {
			a[i], a[j] = a[j], a[i]
		}
	}
	for half := 1; half < n; half <<= 1 {
		w := tab[half : half*2]
		for i := 0; i < n; i += half << 1 {
			p := a[i : i+half : i+half]
			q := a[i+half : i+half*2 : i+half*2]
			for j := 0; j < half; j++ {
				v := m.mul(q[j], w[j])
				u := p[j]
				p[j] = nttAdd(u, v)
				q[j] = nttSub(u, v)
			}
		}
	}
}

func nttColsGo(m *mont, d []uint64, n, width int, tab []uint64, br []int, tmp []uint64) {
	for i, j := range br {
		if j > i {
			ri := d[i*width : i*width+width]
			rj := d[j*width : j*width+width]
			copy(tmp[:width], ri)
			copy(ri, rj)
			copy(rj, tmp[:width])
		}
	}
	for half := 1; half < n; half <<= 1 {
		for i := 0; i < n; i += half << 1 {
			for j := 0; j < half; j++ {
				wj := tab[half+j]
				p := d[(i+j)*width : (i+j)*width+width]
				q := d[(i+j+half)*width : (i+j+half)*width+width]
				for c := range p {
					v := m.mul(q[c], wj)
					u := p[c]
					p[c] = nttAdd(u, v)
					q[c] = nttSub(u, v)
				}
			}
		}
	}
}

func matchExactGo(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if cn < 1 || cn > 4 || step < cn || tw < 1 || th < 1 || tw > iw || th > ih ||
		istride < iw*step || tstride < tw*step {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d, cn=%d step=%d)", tw, th, iw, ih, cn, step))
	}
	threads = clampThreads(threads)
	rw, rh := iw-tw+1, ih-th+1
	res := result
	if res == nil {
		res = make([]float32, rw*rh)
	}
	corrD := make([]float64, rw*rh)

	fp := newGoPlan(tw, th, rw, rh)
	m := newMont()
	g := m.generator()
	fwdW := makeNttTab(&m, g, fp.dftW, false)
	invW := makeNttTab(&m, g, fp.dftW, true)
	fwdH, invH := fwdW, invW
	if fp.dftH != fp.dftW {
		fwdH = makeNttTab(&m, g, fp.dftH, false)
		invH = makeNttTab(&m, g, fp.dftH, true)
	}
	var lut [256]uint64
	for i := range lut {
		lut[i] = m.toMont(uint64(i))
	}

	specN := fp.dftH * fp.dftW
	// reversed template spectra, pre-scaled by (dftW*dftH)^-1 mod p
	tspec := make([]uint64, cn*specN)
	z0 := make([]uint64, fp.dftW)
	ninv := m.pow(m.toMont(uint64(fp.dftW)*uint64(fp.dftH)), nttP-2)
	for k := 0; k < cn; k++ {
		ts := tspec[k*specN : (k+1)*specN]
		for y := 0; y < fp.dftH; y++ {
			row := ts[y*fp.dftW : (y+1)*fp.dftW]
			if y >= th {
				tail := ts[y*fp.dftW:]
				for i := range tail {
					tail[i] = 0
				}
				break
			}
			src := tpl[(th-1-y)*tstride+k:]
			for x := 0; x < tw; x++ {
				row[x] = lut[src[(tw-1-x)*step]]
			}
			for x := tw; x < fp.dftW; x++ {
				row[x] = 0
			}
			nttGo(&m, row, fwdW, fp.brW)
		}
		nttColsGo(&m, ts, fp.dftH, fp.dftW, fwdH, fp.brH, z0)
		for i := range ts {
			ts[i] = m.mul(ts[i], ninv)
		}
	}

	ntx := (rw + fp.blockW - 1) / fp.blockW
	nty := (rh + fp.blockH - 1) / fp.blockH
	ntiles := ntx * nty
	nw := threads
	if nw > ntiles {
		nw = ntiles
	}
	runParallel(nw, func(w int) {
		spec := make([]uint64, specN)
		z := make([]uint64, fp.dftW)
		for t := w; t < ntiles; t += nw {
			x0, y0 := (t%ntx)*fp.blockW, (t/ntx)*fp.blockH
			bw, bh := min(fp.blockW, rw-x0), min(fp.blockH, rh-y0)
			loadW, loadH := bw+tw-1, bh+th-1
			for k := 0; k < cn; k++ {
				for r := 0; r < fp.dftH; r++ {
					row := spec[r*fp.dftW : (r+1)*fp.dftW]
					if r >= loadH {
						tail := spec[r*fp.dftW:]
						for i := range tail {
							tail[i] = 0
						}
						break
					}
					src := img[(y0+r)*istride+x0*step+k:]
					for x := 0; x < loadW; x++ {
						row[x] = lut[src[x*step]]
					}
					for x := loadW; x < fp.dftW; x++ {
						row[x] = 0
					}
					nttGo(&m, row, fwdW, fp.brW)
				}
				nttColsGo(&m, spec, fp.dftH, fp.dftW, fwdH, fp.brH, z)
				ts := tspec[k*specN : (k+1)*specN]
				for i := range spec {
					spec[i] = m.mul(spec[i], ts[i])
				}
				nttColsGo(&m, spec, fp.dftH, fp.dftW, invH, fp.brH, z)
				for r := 0; r < bh; r++ {
					row := spec[(r+th-1)*fp.dftW : (r+th)*fp.dftW]
					nttGo(&m, row, invW, fp.brW)
					o := corrD[(y0+r)*rw+x0:]
					if k == 0 {
						for x := 0; x < bw; x++ {
							o[x] = float64(m.fromMont(row[x+tw-1]))
						}
					} else {
						for x := 0; x < bw; x++ {
							o[x] += float64(m.fromMont(row[x+tw-1]))
						}
					}
				}
			}
		}
	})

	return normalizeParallelGo(img, istride, iw, tpl, tstride, tw, th, cn, step, rw, rh, threads, nil, corrD, res)
}

// ----------------------------------------------------------- entrypoint --

func matchU8Go(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step, threads int, result []float32) (float32, int, int, float32, int, int) {
	if cn < 1 || cn > 4 || step < cn || tw < 1 || th < 1 || tw > iw || th > ih ||
		istride < iw*step || tstride < tw*step {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d, cn=%d step=%d)", tw, th, iw, ih, cn, step))
	}
	threads = clampThreads(threads)
	rw, rh := iw-tw+1, ih-th+1
	res := result
	if res == nil {
		res = make([]float32, rw*rh)
	}
	p := newGoPlan(tw, th, rw, rh)
	crossCorrGo(img, istride, tpl, tstride, step, cn, tw, th, rw, rh, threads, p, res)
	return normalizeParallelGo(img, istride, iw, tpl, tstride, tw, th, cn, step, rw, rh, threads, res, nil, res)
}
