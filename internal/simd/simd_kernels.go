//go:build amd64 && gc && !purego

package simd

// Kernel declarations for the amd64 (AVX2) implementation. Each vector
// lane executes exactly the scalar op sequence of the generic Go loops
// (individual IEEE single-rounded multiplies and adds — never a fused
// multiply-add), so results are bit-identical to the scalar code on
// every input; the asm is a pure speedup, not a numeric variant. Every
// other architecture (arm64 included, by owner decision — the NEON twin
// kernels were deleted for code-size reasons) runs the scalar loops.

// FFTStagesR4 runs the complete radix-4 cascade over one bit-reversed
// row (len a power of two >= 8): the odd-log2 head stage (pure adds),
// the h=1 in-register stage (lanes 1..3 through the w=(1,0) multiply
// idiom, lane 0 untouched exactly like the scalar), then h=2/h>=4
// stages — per element exactly the scalar fftR4 sequence, twiddle
// triplets from the same tri table, inverse via exact wi-lane sign
// flips. The caller performs the swap-pair permutation. No FMA.
//
//go:noescape
func FFTStagesR4(a []complex64, tri []complex64, inverse bool)

// FFTColsR4 applies one radix-4 column stage to the row quad
// (r0, r1, r2, r3) = rows (r, r+h, r+2h, r+3h): per element exactly the
// scalar colsR4Pass sequence — tb = r2*w1, tc = r1*w2, td = r3*w3 with
// the shared two-products-one-addsub multiply idiom (twiddles arrive
// already direction-adjusted), plain-add combines, and the -+i rotation
// as an exact swap + sign-bit xor picked by inverse. No FMA anywhere
// (7.2 contract: the fma32 scalar twin measured 3.8x slower, so fusion
// is banned to keep asm<->purego bit-identity). All rows share r0's
// length; the kernel finishes sub-vector tails itself.
//
//go:noescape
func FFTColsR4(r0, r1, r2, r3 []complex64, w1, w2, w3 complex64, inverse bool)

// FFTColsHead is the odd-log2 head stage on a row pair: p,q = p+q, p-q.
// Plain single-rounded adds, no twiddles — exactly the scalar
// colsR4Head loop.
//
//go:noescape
func FFTColsHead(p, q []complex64)

// MulConj computes spec *= conj(tspec) element-wise.
//
//go:noescape
func MulConj(spec, tspec []complex64)

// NormRow evaluates the TM_CCOEFF_NORMED tail over one chunk of n result
// elements from three float32 lanes at wt, each stride floats apart:
// lane0 (the exact integer cross Σ_k wndSum_k·templSum_k for cn ≥ 3, or
// the raw window sum for cn = 1 with the template mean folded into
// numScale), idiff (area·wndSum2 − Σ_k wndSum_k², the exact-integer
// variance numerator, ≥ 0) and s2 (the raw window square sum) — the
// channel count is folded into the lanes and constants, so one kernel
// serves every cn. Per element: num = corr − lane0·numScale;
// diff2 = idiff·varScale; lim = min(eps·s2, 0.5); den = diff2 > lim ?
// sqrt(diff2)·templNorm : 0; then OpenCV's guard ladder (|num| < den →
// num/den; < den·1.125 → ±1; else 0). Every vector lane executes the
// scalar normOne op sequence with correctly rounded float32
// mul/sub/sqrt/div and exact predicates/bit-selects — no FMA (fusing
// here would break asm↔scalar bit-identity; declined by design, see the
// Phase 7 ledger). n must be a multiple of 8 (the caller finishes the
// tail in Go).
//
//go:noescape
func NormRow(rrow []float32, crow []float32, wt *float32, stride, n int,
	numScale, varScale, eps, templNorm float32)

// PackRows2 fills z[i] = complex(float32(ra[i*step]), float32(rb[i*step]))
// for i < len(z); step must be 1 or 4. uint8->float32 conversions are
// exact, so this is pure data movement. The rows only need the elements
// actually addressed (len >= (len(z)-1)*step+1): the stride-4 gathers
// clamp their wide loads to len(ra)/len(rb), never reading past the
// slice even when the image buffer ends at a page boundary.
//
//go:noescape
func PackRows2(z []complex64, ra, rb []uint8, step int)

// PackRows1 is PackRows2 with a zero imaginary row.
//
//go:noescape
func PackRows1(z []complex64, ra []uint8, step int)

// Untangle splits the packed two-real-rows spectrum: for k in [k0, k1),
// with zk = z[k] and zn = z[n-k] (k >= 1 so the index never wraps),
//
//	sa[k] = (0.5*(zk.re+zn.re), 0.5*(zk.im-zn.im))
//	sb[k] = (0.5*(zk.im+zn.im), 0.5*(zn.re-zk.re))
//
// identical single-rounded ops to the scalar loop.
//
//go:noescape
func Untangle(sa, sb, z []complex64, n, k0, k1 int)

// CombineLow rebuilds z[k] = (sa[k].re - sb[k].im, sa[k].im + sb[k].re)
// for k < len(zlo); CombineHigh fills z[k] = (sa[m].re + sb[m].im,
// sb[m].re - sa[m].im) with m = n-k for k in [hw, n).
//
//go:noescape
func CombineLow(z, sa, sb []complex64)

//go:noescape
func CombineHigh(z, sa, sb []complex64, n, hw int)

// EmitRe stores (add=false) or accumulates (add=true) the real parts of z
// into dst; EmitIm does the same with the imaginary parts. Exact float32
// adds in ascending order, one per element, like the scalar loops.
//
//go:noescape
func EmitRe(dst []float32, z []complex64, add bool)

//go:noescape
func EmitIm(dst []float32, z []complex64, add bool)

// MinMaxRow scans row (len a nonzero multiple of 8, < 2^31) and returns
// the minimum and maximum with OpenCV's first-occurrence semantics: the
// index of the first element equal to each extremum. Comparisons are
// exact predicates (ordered, so NaNs are never selected, like the scalar
// v<min / v>max tests); ties across vector lanes are broken by index, so
// the result is the lexicographic (value, index) extremum — precisely
// what the scalar scan yields.
//
//go:noescape
func MinMaxRow(row []float32) (minV, maxV float32, minI, maxI int)

// RGBAToGray converts len(dst) interleaved RGBA pixels (len a multiple
// of 8; src holds at least 4*len(dst) bytes) with OpenCV's fixed-point
// BT.601 weights (4899*R + 9617*G + 1868*B + 8192) >> 14. Exact integer
// arithmetic, identical to the scalar loop on every input.
//
//go:noescape
func RGBAToGray(dst, src []uint8)

// SlideCols1 advances the single-channel sliding column statistics one
// row: colSum[x] += radd[x]-rsub[x], colSum2[x] += radd[x]²-rsub[x]².
// len(colSum) must be a multiple of 8; the byte rows and colSum2 hold at
// least that many elements. Integer arithmetic is exact, so lane order
// is free.
//
//go:noescape
func SlideCols1(colSum []int32, colSum2 []int64, rsub, radd []uint8)

// SlideCols4 is the RGBA-layout variant over len(colSum2) pixels (a
// multiple of 8): all four colSum lanes slide per pixel, and colSum2
// accumulates the summed squared channel deltas of cn (3 ignores alpha,
// 4 includes it) channels. Reads exactly 4*len(colSum2) row bytes.
//
//go:noescape
func SlideCols4(colSum []int32, colSum2 []int64, rsub, radd []uint8, cn int)

// SpillStats1 runs the single-channel normalize spill over len(wt)
// elements (a multiple of 4): element i gets the three float32 lanes
// s0, idiff = area·s2 − s0² and s2 — each through the same
// exact int64 → float64 → float32 double conversion as the scalar spill
// (float64(v) is one correct rounding for |v| < 2^62, float32 of it the
// second) — then the window slides s0 += hi[i]−lo[i], s2 += hi2[i]−lo2[i];
// the advanced sums are returned. All integer arithmetic is exact: the
// products decompose into signed 32×32→64 pieces (the caller gates
// area ≤ 8421504 so window sums stay below 2^31, and th ≤ 32767 so the
// row-delta products fit) and int64 prefix regrouping is free, so output is
// bit-identical to the scalar loop. wt holds the s0 lane; the idiff and
// s2 lanes live stride and 2·stride floats later. lo/hi/lo2/hi2 hold at
// least len(wt) elements.
//
//go:noescape
func SpillStats1(wt []float32, stride int, lo, hi []int32, lo2, hi2 []int64, s0, s2, area int64) (ns0, ns2 int64)
