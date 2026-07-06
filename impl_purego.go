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
