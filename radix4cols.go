package cvmatch

import "github.com/hkloudou/cvmatch/internal/simd"

// Column-pass twin of the radix-4 engine (Phase 7.2). colsR4Go applies
// to every column of a row-major [n x width] slab exactly the op
// sequence fftR4 applies to a 1-D vector — TestColsR4MatchesRowEngine
// pins that bit-for-bit — while keeping fftColsGo's scheduling shape:
// row-swap bit reversal, one memory sweep per pass, team width-chunks
// with runParallel as the inter-pass barrier. Like the row engine it is
// not wired into the pipeline yet; the flip lands with the asm twins
// and the single golden re-record.
//
// Arithmetic note vs the old schedule: fftColsGo runs its odd stage as
// a twiddled radix-2 pass at the top (half = n/2); this schedule runs
// it at the bottom (half = 1) where every twiddle is 1, so the head
// pass is pure adds — one fewer twiddled sweep on odd-log2 heights.

// colsR4Head is the odd-log2 head stage: rows (2i, 2i+1) combine as
// (a+b, a-b) per column — plain single-rounded adds, no twiddles.
func colsR4Head(d []complex64, n, width, c0, c1 int) {
	for r := 0; r < n; r += 2 {
		p := d[r*width+c0 : r*width+c1]
		q := d[(r+1)*width+c0 : (r+1)*width+c1]
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
func colsR4Pass(d []complex64, n, width int, w1, w2, w3 []complex64, inverse bool, h, c0, c1 int) {
	for base := 0; base < n; base += 4 * h {
		for j := 0; j < h; j++ {
			bw, cw, dw := w1[j], w2[j], w3[j]
			if inverse {
				bw = complex(real(bw), -imag(bw))
				cw = complex(real(cw), -imag(cw))
				dw = complex(real(dw), -imag(dw))
			}
			r := base + j
			pA := d[r*width+c0 : r*width+c1]
			pC := d[(r+h)*width+c0 : (r+h)*width+c1]
			pB := d[(r+2*h)*width+c0 : (r+2*h)*width+c1]
			pD := d[(r+3*h)*width+c0 : (r+3*h)*width+c1]
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
// place. tmp must hold at least width elements (the row-swap scratch,
// same convention as fftColsGo).
func colsR4Go(d []complex64, n, width int, t *r4Tab, inverse bool, tmp []complex64, team int) {
	for k := 0; k+1 < len(t.pairs); k += 2 {
		i, j := int(t.pairs[k]), int(t.pairs[k+1])
		ri := d[i*width : i*width+width]
		rj := d[j*width : j*width+width]
		copy(tmp[:width], ri)
		copy(ri, rj)
		copy(rj, tmp[:width])
	}
	h := 1
	if oddLog2(n) {
		colRange(team, width, func(c0, c1 int) { colsR4Head(d, n, width, c0, c1) })
		h = 2
	}
	tri := t.tri
	for ; 4*h <= n; h *= 4 {
		w1, w2, w3 := tri[:h], tri[h:2*h], tri[2*h:3*h]
		tri = tri[3*h:]
		hh := h
		colRange(team, width, func(c0, c1 int) {
			colsR4Pass(d, n, width, w1, w2, w3, inverse, hh, c0, c1)
		})
	}
}
