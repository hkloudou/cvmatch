package cvmatch

import (
	"fmt"
	"image"
	"sync"
	"sync/atomic"
)

// Matcher holds the reusable template-side state for searching one sub
// image repeatedly — the production shape where a fixed template (a
// button, an icon) is hunted across a stream of screenshots. It caches
// everything derived from the sub image: a private pixel copy, the
// exact integer statistics, and — per parent geometry — the FFT plan
// and scaled template spectrum, which one-shot Match/MatchGray rebuild
// on every call. Results are bit-identical to the one-shot functions on
// every input (the cache stores the same values those calls compute).
//
// A Matcher is safe for concurrent Find calls. The spectrum cache is
// keyed by the parent's dimensions (and the effective channel count):
// it builds on the first Find and rebuilds only when those change, so a
// fixed-resolution screen stream hits the cache from the second call
// on. Memory per cached geometry is cn·dftH·(dftW/2+1) complex64s —
// roughly the size of one padded FFT tile per channel.
type Matcher struct {
	pix    []uint8 // private compact copy (stride = w*step)
	w, h   int
	step   int // 4: RGBA (Match semantics), 1: gray (MatchGray semantics)
	aConst bool
	tsum   [4]int64
	varSum int64
	mu     sync.Mutex
	geom   atomic.Pointer[matcherGeom]
}

type matcherGeom struct {
	rw, rh, cn int
	p          *goPlan
	set        *tspecSet
}

// NewMatcher prepares sub for repeated color searches: Find(parent) is
// bit-identical to Match(parent, sub). Panics like Match on an empty
// image or a template above the exact-statistics bound.
func NewMatcher(sub image.Image) *Matcher {
	pix, stride, w, h, owned := toRGBA(sub)
	m := newMatcher(pix, stride, w, h, 4)
	m.aConst = alphaConst(m.pix, m.w*4, m.w, m.h)
	if owned {
		bytePool.put(pix)
	}
	return m
}

// NewGrayMatcher prepares sub for repeated grayscale searches:
// Find(parent) is bit-identical to MatchGray(parent, sub).
func NewGrayMatcher(sub image.Image) *Matcher {
	pix, stride, w, h, owned := toGray(sub)
	m := newMatcher(pix, stride, w, h, 1)
	if owned {
		bytePool.put(pix)
	}
	return m
}

func newMatcher(pix []uint8, stride, w, h, step int) *Matcher {
	if int64(w)*int64(h) > statsCap(step) { // step==cn upper bound (4 or 1)
		panic(fmt.Sprintf("cvmatch: template area %dx%d exceeds the exact-statistics bound", w, h))
	}
	m := &Matcher{pix: make([]uint8, w*h*step), w: w, h: h, step: step}
	for y := 0; y < h; y++ {
		copy(m.pix[y*w*step:(y+1)*w*step], pix[y*stride:y*stride+w*step])
	}
	m.tsum, m.varSum = templStats(m.pix, w*step, w, h, step, step)
	return m
}

// Find searches parent for the prepared template and returns, in order,
// the minimum match value and its X/Y position, then the maximum and
// its X/Y position — exactly Match's (or MatchGray's) tuple, bit for
// bit. Panics like the one-shot functions if the template is larger
// than parent.
func (m *Matcher) Find(parent image.Image) (float32, int, int, float32, int, int) {
	var pPix []uint8
	var pStride, pw, ph int
	var pOwned bool
	if m.step == 1 {
		pPix, pStride, pw, ph, pOwned = toGray(parent)
	} else {
		pPix, pStride, pw, ph, pOwned = toRGBA(parent)
	}
	defer func() {
		if pOwned {
			bytePool.put(pPix)
		}
	}()
	if m.w > pw || m.h > ph {
		panic(fmt.Sprintf("cvmatch: bad match arguments (%dx%d in %dx%d)", m.w, m.h, pw, ph))
	}
	if m.varSum == 0 { // exactly flat template: scores 1 everywhere
		return 1, 0, 0, 1, 0, 0
	}
	cn := m.step // gray: 1
	if m.step == 4 {
		cn = 4
		// The constant-channel skip needs BOTH alpha planes constant —
		// the template half is precomputed, the parent scan runs per
		// frame exactly like Match.
		if m.aConst && alphaConst(pPix, pStride, pw, ph) {
			cn = 3
		}
	}
	rw, rh := pw-m.w+1, ph-m.h+1
	nthreads := threads()
	g := m.geom.Load()
	if g == nil || g.rw != rw || g.rh != rh || g.cn != cn {
		m.mu.Lock()
		if g = m.geom.Load(); g == nil || g.rw != rw || g.rh != rh || g.cn != cn {
			p := newGoPlan(m.w, m.h, rw, rh)
			set := buildTSpecSet(m.pix, m.w*m.step, m.step, cn, m.w, m.h, rh, m.h, nthreads, p, false)
			g = &matcherGeom{rw: rw, rh: rh, cn: cn, p: p, set: set}
			m.geom.Store(g)
		}
		m.mu.Unlock()
	}
	res := f32Pool.get(rw * rh)
	defer f32Pool.put(res)
	crossCorrGo(pPix, pStride, m.step, cn, m.w, m.h, rw, rh, nthreads, g.p, g.set, res)
	return normalizeParallelGo(pPix, pStride, pw, m.w, m.h, cn, m.step, rw, rh, nthreads, &m.tsum, m.varSum, res, res)
}
