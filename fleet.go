package cvmatch

import (
	"fmt"
	"image"
	"sync"
	"sync/atomic"
)

// Result is one matcher's answer from Fleet.FindAll — Match's six-tuple
// as a struct, in the fleet's construction order.
type Result struct {
	MinV       float32
	MinX, MinY int
	MaxV       float32
	MaxX, MaxY int
}

// Fleet batches many prepared Matchers against one parent frame — the
// production shape where hundreds of templates are hunted across every
// screenshot. One FindAll converts the parent once, scans its alpha
// once, and runs the parent-side forward FFT of every tile ONCE into a
// shared frame-spectrum cache; each matcher then pays only its own
// tail (conjugate multiply, inverse transform, emit, normalization,
// extremum scan). All members share a single tile geometry planned for
// the largest template, so per-member results can deviate from solo
// Find at the usual plan-change tolerance (~1e-5 score class, same
// contract as the tile argmin itself); a single-member fleet plans
// identically to Find and reproduces it bit for bit. For a fixed
// parent size the shared plan and every member's spectrum build once,
// on the first frame.
//
// FindAll is safe for concurrent use. Frame-cache memory during a call
// is cnMax·Σ_tiles·specN complex64s (on the order of the padded parent
// area per channel); each member's cached spectrum adds cn·specN as
// with a solo Matcher.
type Fleet struct {
	ms           []*Matcher
	step         int
	twMax, thMax int
	mu           sync.Mutex
	geom         atomic.Pointer[fleetGeom]
}

type fleetGeom struct {
	pw, ph int
	pConst bool // parent alpha constant (color mode)
	p      *goPlan
	cns    []int
	sets   []*tspecSet
	cnMax  int
}

// NewFleet groups prepared Matchers for batched searching. All members
// must share one mode (all NewMatcher or all NewGrayMatcher); panics on
// an empty or mixed fleet.
func NewFleet(ms ...*Matcher) *Fleet {
	if len(ms) == 0 {
		panic("cvmatch: empty fleet")
	}
	f := &Fleet{ms: ms, step: ms[0].step}
	for _, m := range ms {
		if m.step != f.step {
			panic("cvmatch: fleet mixes color and gray matchers")
		}
		f.twMax = max(f.twMax, m.w)
		f.thMax = max(f.thMax, m.h)
	}
	return f
}

// FindAll searches parent for every member and returns their results in
// construction order. Panics like Find if any template is larger than
// parent.
func (f *Fleet) FindAll(parent image.Image) []Result {
	var pPix []uint8
	var pStride, pw, ph int
	var pOwned bool
	if f.step == 1 {
		pPix, pStride, pw, ph, pOwned = toGray(parent)
	} else {
		pPix, pStride, pw, ph, pOwned = toRGBA(parent)
	}
	defer func() {
		if pOwned {
			bytePool.put(pPix)
		}
	}()
	if f.twMax > pw || f.thMax > ph {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d)", f.twMax, f.thMax, pw, ph))
	}
	pConst := f.step == 4 && alphaConst(pPix, pStride, pw, ph)
	nthreads := threads()
	g := f.geom.Load()
	if g == nil || g.pw != pw || g.ph != ph || g.pConst != pConst {
		f.mu.Lock()
		if g = f.geom.Load(); g == nil || g.pw != pw || g.ph != ph || g.pConst != pConst {
			g = f.buildGeom(pw, ph, pConst, nthreads)
			f.geom.Store(g)
		}
		f.mu.Unlock()
	}
	p := g.p
	rwF, rhF := pw-f.twMax+1, ph-f.thMax+1
	ntx := (rwF + p.blockW - 1) / p.blockW
	nty := (rhF + p.blockH - 1) / p.blockH
	ntiles := ntx * nty
	lastY0 := ((rhF - 1) / p.blockH) * p.blockH

	// Per-tile plan and frame-slab offsets: last-band tiles may run the
	// shrunk transform height (7.4-lite, decided once for the fleet from
	// thMax so every member shares the band geometry).
	specN := p.dftH * p.hw
	tileP := func(y0 int) *goPlan {
		if y0 == lastY0 && g.sets[0].p2 != nil {
			return g.sets[0].p2
		}
		return p
	}
	off := make([]int, ntiles+1)
	for t := 0; t < ntiles; t++ {
		bp := tileP(t / ntx * p.blockH)
		off[t+1] = off[t] + g.cnMax*bp.dftH*bp.hw
	}
	frame := cplxPool.get(off[ntiles])

	// Phase 1: every tile's parent spectrum, once for the whole fleet.
	team := 1
	if nw := min(nthreads, ntiles); teamEligible(p, g.cnMax) {
		team = nthreads / nw
	}
	var next atomic.Int64
	runParallel(min(nthreads, ntiles), func(int) {
		z := cplxPool.get(p.dftW)
		defer cplxPool.put(z)
		for {
			t := int(next.Add(1) - 1)
			if t >= ntiles {
				return
			}
			x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
			bwF, bhF := min(p.blockW, rwF-x0), min(p.blockH, rhF-y0)
			bp := tileP(y0)
			sn := bp.dftH * bp.hw
			for k := 0; k < g.cnMax; k++ {
				blockForwardGo(pPix[k:], pStride, f.step, x0, y0,
					bwF+f.twMax-1, bhF+f.thMax-1, bp, frame[off[t]+k*sn:off[t]+(k+1)*sn], z, team)
			}
		}
	})

	// Phase 2: each member's tail — conjugate multiply against its own
	// spectrum (on a copy, the frame cache is read-only here), inverse,
	// emit into its own map, normalize, scan. Members are independent
	// given the frame cache, and every inner op is scheduling-free, so
	// results are bit-identical for any worker count or member order.
	results := make([]Result, len(f.ms))
	outer := min(nthreads, len(f.ms))
	inner := max(nthreads/outer, 1)
	runParallel(outer, func(w int) {
		spec := cplxPool.get(specN)
		z := cplxPool.get(p.dftW)
		defer cplxPool.put(spec)
		defer cplxPool.put(z)
		for mi := w; mi < len(f.ms); mi += outer {
			m := f.ms[mi]
			if m.varSum == 0 {
				results[mi] = Result{1, 0, 0, 1, 0, 0}
				continue
			}
			cn, set := g.cns[mi], g.sets[mi]
			rw, rh := pw-m.w+1, ph-m.h+1
			res := f32Pool.get(rw * rh)
			for t := 0; t < ntiles; t++ {
				x0, y0 := (t%ntx)*p.blockW, (t/ntx)*p.blockH
				bp, ts := p, set.spec
				if y0 == lastY0 && set.p2 != nil {
					bp, ts = set.p2, set.spec2
				}
				sn := bp.dftH * bp.hw
				// Interior tiles emit the shared block width for every
				// member; the last tile per axis extends to the member's
				// own result edge — exactly the extra valid width its
				// smaller template earns from the shared loads.
				bw, bh := p.blockW, p.blockH
				if t%ntx == ntx-1 {
					bw = rw - x0
				}
				if t/ntx == nty-1 {
					bh = rh - y0
				}
				for k := 0; k < cn; k++ {
					copy(spec[:sn], frame[off[t]+k*sn:off[t]+(k+1)*sn])
					mulConjGo(spec[:sn], ts[k*sn:(k+1)*sn])
					blockInverseEmitGo(bp, spec, z, res, rw, x0, y0, bw, bh, k > 0, inner)
				}
			}
			r := Result{}
			r.MinV, r.MinX, r.MinY, r.MaxV, r.MaxX, r.MaxY = normalizeParallelGo(
				pPix, pStride, pw, m.w, m.h, cn, f.step, rw, rh, inner,
				&m.tsum, m.varSum, res, res)
			results[mi] = r
			f32Pool.put(res)
		}
	})
	cplxPool.put(frame)
	return results
}

// buildGeom plans the shared tile geometry for one parent size and
// builds every member's spectrum on it. Called under f.mu.
func (f *Fleet) buildGeom(pw, ph int, pConst bool, nthreads int) *fleetGeom {
	rwF, rhF := pw-f.twMax+1, ph-f.thMax+1
	p := newGoPlan(f.twMax, f.thMax, rwF, rhF)
	g := &fleetGeom{pw: pw, ph: ph, pConst: pConst, p: p,
		cns: make([]int, len(f.ms)), sets: make([]*tspecSet, len(f.ms))}
	for i, m := range f.ms {
		cn := m.step
		if m.step == 4 {
			cn = 4
			if m.aConst && pConst {
				cn = 3
			}
		}
		g.cns[i] = cn
		g.cnMax = max(g.cnMax, cn)
		// shrinkTH = fleet thMax: every member's shrunk-band spectrum
		// lands on the one band geometry the shared forward pass uses.
		g.sets[i] = buildTSpecSet(m.pix, m.w*m.step, m.step, cn, m.w, m.h,
			rhF, f.thMax, nthreads, p, false)
	}
	return g
}
