package cvmatch

// Pure-Go port of cvmatch.c, selected automatically when cgo is off (see
// impl_nocgo.go). It runs the same algorithm — block-FFT cross-correlation
// with OpenCV's tile heuristic, sliding-column-sum normalization with
// OpenCV's exact guards, fused minMaxLoc — and is always compiled so tests
// can compare it against the C core when cgo is on.

import (
	"fmt"
	"math"
)

func nextPow2(v int) int {
	n := 1
	for n < v {
		n <<= 1
	}
	return n
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
	spec, tspec    []complex64
	ztmp           []complex64
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
	p.spec = make([]complex64, dftH*p.hw)
	p.tspec = make([]complex64, dftH*p.hw)
	p.ztmp = make([]complex64, dftW)
	return p
}

// blockForwardGo: real 2D forward DFT of one channel of a uint8 block, two
// real rows packed per complex FFT.
func blockForwardGo(chanBase []uint8, stride, step, x0, y0, loadW, loadH int, p *goPlan, spec []complex64) {
	n, hw, mask := p.dftW, p.hw, p.dftW-1
	z := p.ztmp
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
	fftColsGo(spec, p.dftH, hw, p.twH, p.brH, false, p.ztmp)
}

func mulConjGo(spec, tspec []complex64) {
	for i, a := range spec {
		b := tspec[i]
		spec[i] = complex(real(a)*real(b)+imag(a)*imag(b), imag(a)*real(b)-real(a)*imag(b))
	}
}

func blockInverseEmitGo(p *goPlan, spec []complex64, res []float32, rw, x0, y0, bw, bh int, add bool) {
	fftColsGo(spec, p.dftH, p.hw, p.twH, p.brH, true, p.ztmp)
	n, hw := p.dftW, p.hw
	z := p.ztmp
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

func crossCorrChannelGo(img []uint8, istride int, tpl []uint8, tstride, step, tw, th, rw, rh int, p *goPlan, add bool, result []float32) {
	blockForwardGo(tpl, tstride, step, 0, 0, tw, th, p, p.tspec)
	scale := complex(1/(float32(p.dftW)*float32(p.dftH)), 0)
	for i := range p.tspec {
		p.tspec[i] *= scale
	}
	for y0 := 0; y0 < rh; y0 += p.blockH {
		bh := min(p.blockH, rh-y0)
		for x0 := 0; x0 < rw; x0 += p.blockW {
			bw := min(p.blockW, rw-x0)
			blockForwardGo(img, istride, step, x0, y0, bw+tw-1, bh+th-1, p, p.spec)
			mulConjGo(p.spec, p.tspec)
			blockInverseEmitGo(p, p.spec, result, rw, x0, y0, bw, bh, add)
		}
	}
}

// ------------------------------------------------- normalization + scan --

func normalizeAndScanGo(img []uint8, istride, iw int, tpl []uint8, tstride, tw, th, cn, step, rw, rh int, result []float32) (minV float64, minX, minY int, maxV float64, maxX, maxY int) {
	area := float64(tw) * float64(th)
	invArea := 1 / area
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
	templNorm = math.Sqrt(templNorm * area)

	colSum := make([]float64, iw*cn)
	colSum2 := make([]float64, iw)
	for y := 0; y < th; y++ {
		row := img[y*istride:]
		for x := 0; x < iw; x++ {
			for k := 0; k < cn; k++ {
				v := float64(row[x*step+k])
				colSum[x*cn+k] += v
				colSum2[x] += v * v
			}
		}
	}

	minv, maxv := math.MaxFloat64, -math.MaxFloat64
	var minx, miny, maxx, maxy int
	const eps = 10.0 * 1.1920929e-07 // 10*FLT_EPSILON

	for y := 0; ; y++ {
		var s [4]float64
		s2 := 0.0
		for x := 0; x < tw; x++ {
			for k := 0; k < cn; k++ {
				s[k] += colSum[x*cn+k]
			}
			s2 += colSum2[x]
		}
		rrow := result[y*rw : y*rw+rw]
		for x := 0; ; x++ {
			num := float64(rrow[x])
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
			if float64(v) < minv {
				minv, minx, miny = float64(v), x, y
			}
			if float64(v) > maxv {
				maxv, maxx, maxy = float64(v), x, y
			}
			if x+1 >= rw {
				break
			}
			for k := 0; k < cn; k++ {
				s[k] += colSum[(x+tw)*cn+k] - colSum[x*cn+k]
			}
			s2 += colSum2[x+tw] - colSum2[x]
		}
		if y+1 >= rh {
			break
		}
		sub := img[y*istride:]
		addr := img[(y+th)*istride:]
		for x := 0; x < iw; x++ {
			for k := 0; k < cn; k++ {
				a := float64(addr[x*step+k])
				b := float64(sub[x*step+k])
				colSum[x*cn+k] += a - b
				colSum2[x] += a*a - b*b
			}
		}
	}
	return minv, minx, miny, maxv, maxx, maxy
}

// ----------------------------------------------------------- entrypoint --

func matchU8Go(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step int, result []float32) (float32, int, int, float32, int, int) {
	if cn < 1 || cn > 4 || step < cn || tw < 1 || th < 1 || tw > iw || th > ih ||
		istride < iw*step || tstride < tw*step {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d, cn=%d step=%d)", tw, th, iw, ih, cn, step))
	}
	rw, rh := iw-tw+1, ih-th+1
	res := result
	if res == nil {
		res = make([]float32, rw*rh)
	}
	p := newGoPlan(tw, th, rw, rh)
	for k := 0; k < cn; k++ {
		crossCorrChannelGo(img[k:], istride, tpl[k:], tstride, step, tw, th, rw, rh, p, k > 0, res)
	}
	minV, minX, minY, maxV, maxX, maxY := normalizeAndScanGo(img, istride, iw, tpl, tstride, tw, th, cn, step, rw, rh, res)
	return float32(minV), minX, minY, float32(maxV), maxX, maxY
}
