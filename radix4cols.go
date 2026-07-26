package cvmatch

import "github.com/hkloudou/cvmatch/internal/simd"

// Column-pass twin of the radix-4 engine (Phase 7.2). colsR4Go applies
// to every column of a row-major [n x width] slab exactly the op
// sequence fftR4 applies to a 1-D vector — TestColsR4MatchesRowEngine
// pins that bit-for-bit — while keeping fftColsGo's scheduling shape:
// one memory sweep per pass, team width-chunks with runParallel as the
// inter-pass barrier. This IS the pipeline's column engine
// (blockForwardGo/blockInverseEmitGo call it).
//
// Unlike the row engine there is no physical bit-reversal here (Phase
// 9: the row-swap sweeps were ~7% of timed work on big tiles). The
// cascade wants logical row L at slot brev[L]; forward callers get that
// for free by *writing* their rows at brev slots (natural cascade
// indexing, rmap == nil, output lands natural). The inverse consumes
// MulConj's natural layout unmoved by conjugating every row access with
// the involution instead (rmap = brev): logical slot s is read/written
// at physical row rmap[s], which leaves the spatial row y at slot
// brev[y] for the emit to read back through one lookup. Both modes run
// the identical op sequence on identical values — pure storage
// relabeling, bit-identical output, asm kernels untouched (they only
// ever see row slices).
//
// Arithmetic note vs the retired radix-2 column passes: they ran the
// odd stage as a twiddled pass at the top (half = n/2); this schedule
// runs it at the bottom (half = 1) where every twiddle is 1, so the
// head pass is pure adds — one fewer twiddled sweep on odd-log2
// heights. Measured at the flip: +14.5% asm / +18.7% purego on the
// 512x256 pipeline-shaped slab.

// colsR4Head is the odd-log2 head stage: rows (2i, 2i+1) combine as
// (a+b, a-b) per column — plain single-rounded adds, no twiddles.
func colsR4Head(d []complex64, n, width, c0, c1 int, rmap []int32) {
	for r := 0; r < n; r += 2 {
		i0, i1 := r, r+1
		if rmap != nil {
			i0, i1 = int(rmap[i0]), int(rmap[i1])
		}
		p := d[i0*width+c0 : i0*width+c1]
		q := d[i1*width+c0 : i1*width+c1]
		if simd.Enabled {
			simd.FFTColsHead(p, q)
			continue
		}
		for c := range p {
			u, v := p[c], q[c]
			p[c] = complex(real(u)+real(v), imag(u)+imag(v))
			q[c] = complex(real(u)-real(v), imag(u)-imag(v))
		}
	}
}

// colsR4Pass applies one radix-4 stage (quarter size h) to columns
// [c0, c1). Input roles follow the bit-reversal layout: the q=1
// sub-transform lives at row offset 2h, q=2 at offset h (see the
// contract in radix4.go).
func colsR4Pass(d []complex64, n, width int, w1, w2, w3 []complex64, inverse bool, h, c0, c1 int, rmap []int32) {
	for base := 0; base < n; base += 4 * h {
		for j := 0; j < h; j++ {
			bw, cw, dw := w1[j], w2[j], w3[j]
			if inverse {
				bw = complex(real(bw), -imag(bw))
				cw = complex(real(cw), -imag(cw))
				dw = complex(real(dw), -imag(dw))
			}
			r := base + j
			i0, i1, i2, i3 := r, r+h, r+2*h, r+3*h
			if rmap != nil {
				i0, i1, i2, i3 = int(rmap[i0]), int(rmap[i1]), int(rmap[i2]), int(rmap[i3])
			}
			pA := d[i0*width+c0 : i0*width+c1]
			pC := d[i1*width+c0 : i1*width+c1]
			pB := d[i2*width+c0 : i2*width+c1]
			pD := d[i3*width+c0 : i3*width+c1]
			if simd.Enabled {
				simd.FFTColsR4(pA, pC, pB, pD, bw, cw, dw, inverse)
				continue
			}
			for c := range pA {
				av := pA[c]
				tb := mulPlain(pB[c], bw)
				tc := mulPlain(pC[c], cw)
				td := mulPlain(pD[c], dw)
				s0 := complex(real(av)+real(tc), imag(av)+imag(tc))
				s1 := complex(real(av)-real(tc), imag(av)-imag(tc))
				s2 := complex(real(tb)+real(td), imag(tb)+imag(td))
				s3 := complex(real(tb)-real(td), imag(tb)-imag(td))
				var rr, ri float32
				if inverse {
					rr, ri = -imag(s3), real(s3)
				} else {
					rr, ri = imag(s3), -real(s3)
				}
				pA[c] = complex(real(s0)+real(s2), imag(s0)+imag(s2))
				pC[c] = complex(real(s1)+rr, imag(s1)+ri)
				pB[c] = complex(real(s0)-real(s2), imag(s0)-imag(s2))
				pD[c] = complex(real(s1)-rr, imag(s1)-ri)
			}
		}
	}
}

// colsR4Go transforms the columns of the row-major [n x width] slab in
// place, never moving a row: with rmap == nil the cascade indexes rows
// directly (callers must have placed logical row L at slot brev[L];
// output lands natural), with rmap = brev every access is conjugated by
// the involution (input natural; the transform of logical slot s lands
// at physical row brev[s]).
func colsR4Go(d []complex64, n, width int, tri []complex64, inverse bool, rmap []int32, team int) {
	h := 1
	if oddLog2(n) {
		colRange(team, width, func(c0, c1 int) { colsR4Head(d, n, width, c0, c1, rmap) })
		h = 2
	}
	for ; 4*h <= n; h *= 4 {
		w1, w2, w3 := tri[:h], tri[h:2*h], tri[2*h:3*h]
		tri = tri[3*h:]
		hh := h
		colRange(team, width, func(c0, c1 int) {
			colsR4Pass(d, n, width, w1, w2, w3, inverse, hh, c0, c1, rmap)
		})
	}
}
