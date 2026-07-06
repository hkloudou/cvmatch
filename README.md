# cvmatch

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml) [![Go Reference](https://pkg.go.dev/badge/github.com/hkloudou/cvmatch.svg)](https://pkg.go.dev/github.com/hkloudou/cvmatch) ![CGO optional](https://img.shields.io/badge/CGO-optional-success)

OpenCV-compatible `TM_CCOEFF_NORMED` template matching for Go. No OpenCV, no
bundled multi-megabyte static libraries, no dependencies — and **no required
toolchain**: with cgo the core is **one dependency-free C file** (compiles in
~1 s during `go build`, ~35 KB of native code including threading, AVX2
multi-versioned); with
`CGO_ENABLED=0` a **pure-Go port of the same algorithm** takes over
automatically, so plain cross-compilation just works. `cvmatch.Impl` reports
which core is active ("cgo" / "purego"); tests pin both to identical output.

```go
import "github.com/hkloudou/cvmatch"

// Drop-in replacement for cv2.Match (same signature, same numbers):
minVal, minX, minY, maxVal, maxX, maxY := cvmatch.Match(parent, sub)

// Grayscale fast path (~2.5x faster again; ideal for screenshots/UI
// hunting). Alpha is dropped here by design — the same semantics as
// OpenCV's COLOR_RGBA2GRAY; use Match when alpha carries signal:
minVal, minX, minY, maxVal, maxX, maxY = cvmatch.MatchGray(parent, sub)

// Full response map (find every occurrence above a threshold):
resp, w, h := cvmatch.MatchMap(parent, sub)

```

`Match` panics if an image is empty or `sub` is larger than `parent`, matching
the behaviour of the OpenCV-backed original.

## What it does

`Match` is `matchTemplate(TM_CCOEFF_NORMED)` + `minMaxLoc` in one call:
**find the best match of a small image inside a big one**. Every score is a
Pearson correlation coefficient in `[-1, +1]`: `+1` = perfect match (up to a
brightness/contrast shift), `0` = unrelated, `-1` = perfect *inverse* match
(the window looks like the template with colors inverted). So the answer is
the maximum — `maxVal` close to `1.0` means a confident hit whose top-left
corner is `(maxX, maxY)`. `minVal`/`minLoc` are **not** "the worst spot"
(that would be ≈0); they are the strongest inverse match — useful for
hunting color-flipped patterns (dark-mode icons), ignorable otherwise, and
kept for six-tuple compatibility with cv2/OpenCV, where the SQDIFF methods
treat the *minimum* as best.

`MatchMap` returns the whole score map (`resp[y*w+x]` = score of placing the
template's top-left corner at `(x, y)`), which unlocks what a single best
hit cannot: finding **every** occurrence (threshold + local-maximum
suppression with a radius about half the template size), **ambiguity
checks** (if the second-highest peak nearly ties the highest — three
look-alike buttons — the match is not trustworthy even at `maxVal≈1`), ROI-
restricted search, sub-pixel refinement by fitting the peak's neighborhood,
and heatmap debugging. Below, the green box is drawn at the
location `Match` returned for each template (images live on the
[`assets`](../../tree/assets) branch; rendered by `bench/cmd/annotate`):

| template | found in parent | result |
|---|---|---|
| ![button](../../raw/assets/demo/window1600_button96x32.tpl.png) | ![window](../../raw/assets/demo/window1600_button96x32.jpg) | `maxVal=0.999996` @ (893,614) — the right button among three look-alikes |
| ![fruits template](../../raw/assets/demo/photo_fruits.tpl.png) | ![fruits](../../raw/assets/demo/photo_fruits.jpg) | `maxVal=1.000000` @ (305,210) |
| ![baboon template](../../raw/assets/demo/photo_baboon.tpl.png) | ![baboon](../../raw/assets/demo/photo_baboon.jpg) | `maxVal=1.000000` @ (250,180) |
| ![building template](../../raw/assets/demo/photo_building.tpl.png) | ![building](../../raw/assets/demo/photo_building.jpg) | `maxVal=1.000000` @ (420,240) |
| ![graf template](../../raw/assets/demo/photo_graf1.tpl.png) | ![graf](../../raw/assets/demo/photo_graf1.jpg) | `maxVal=0.999999` @ (350,260) |
| ![starry template](../../raw/assets/demo/photo_starry_night.tpl.png) | ![starry night](../../raw/assets/demo/photo_starry_night.jpg) | `maxVal=0.999999` @ (400,300) |
| ![alpha template](../../raw/assets/demo/noise640_alpha.tpl.png) | ![varying alpha](../../raw/assets/demo/noise640_alpha.png) | `maxVal=1.000000` @ (217,143) — **varying-alpha** scene (PNG, alpha preserved): full 4-channel path, pinned element-wise vs OpenCV |

## Benchmarks

Scenarios cover the real workload — finding a button, a toolbar icon or a
panel inside a rendered desktop window (flat regions, gradients, text,
look-alike widgets) — plus dense-noise worst cases and **real photographs**.

The sample photographs come from
[OpenCV's `samples/data`](https://github.com/opencv/opencv/tree/4.12.0/samples/data)
(Apache-2.0; `bench/testdata/fetch.sh` downloads them for the benchmarks, and
they are mirrored on the [`assets`](../../tree/assets) branch for viewing —
the suite runs with synthetic scenes only when they are absent):

| image | size | template | content |
|---|---|---|---|
| [`fruits.jpg`](../../raw/assets/samples/fruits.jpg) | 512×480 | 80×80 @ (305,210) | fruit bowl, saturated colors |
| [`baboon.jpg`](../../raw/assets/samples/baboon.jpg) | 512×512 | 64×64 @ (250,180) | fur, high-frequency texture |
| [`building.jpg`](../../raw/assets/samples/building.jpg) | 868×600 | 100×100 @ (420,240) | architecture, repeating windows |
| [`graf1.png`](../../raw/assets/samples/graf1.png) | 800×640 | 120×120 @ (350,260) | graffiti wall (VGG dataset) |
| [`starry_night.jpg`](../../raw/assets/samples/starry_night.jpg) | 752×600 | 128×128 @ (400,300) | painting, swirling gradients |

The baseline is **native OpenCV C++** — `bench/cpp/native_bench`, plain C++
linked against a prebuilt static OpenCV 4.12 (the same binary the
`hkloudou/cv2` module bundles), timed **end-to-end per call** for fairness
(from an in-memory RGBA buffer: Mat copy → `matchTemplate` → `minMaxLoc` →
release), best-of-5 — measured against **cvmatch.Match** and
**cvmatch.MatchGray**, all returning the same location and value.

> The Go-wrapper question was measured and settled before retiring the
> wrapper from this suite: `hkloudou/cv2`'s Go API landed within **~0-4%**
> of native C++ on every scene (e.g. 1080p/128: 390.5 ms native vs 392.7 ms
> Go), so all cv2 numbers in the tables below are interchangeable with
> native C++. The cost is OpenCV's own pipeline — 4-channel DFT correlation
> plus full double-precision integral images — which is exactly what
> cvmatch restructures. Single-threaded, cvmatch is **2.0-3.8x faster than
> native OpenCV C++ at identical output values**; with its internal
> bit-identical parallelism on 4 cores it is **1.9-9.9x faster (7-9x on the
> big scenes), and 10-16x in grayscale mode** (OpenCV's matchTemplate path
> cannot use extra cores at all).

### Fairness rules

- **End-to-end timing everywhere.** Every implementation is timed over the
  complete per-call path starting from an in-memory RGBA frame — image/Mat
  conversion, matching, min/max scan, cleanup. No implementation gets to keep
  pre-converted inputs or reuse buffers across calls.
- **Byte-identical inputs.** `bench/cmd/dumpscenes` exports the exact pixel
  buffers the Go benchmarks use; the C++ benchmark consumes those dumps.
- **Same OpenCV binary.** The native benchmark links the very static
  archives shipped in the `cv2` module (not a rebuilt variant), so
  Go-vs-native comparisons cannot be skewed by build flags. A distro-built
  OpenCV (Ubuntu 24.04, 4.6.0) was also measured and lands within ±7% of the
  bundled build on every scene.
- **Threading disclosed, and both sides got the cores.** cvmatch.Match
  parallelizes one call internally with provably bit-identical output
  (`TestThreadsBitIdentical*`); OpenCV's matchTemplate path is
  single-threaded by design, so before retiring the wrapper this suite also
  measured **OpenCV + caller-side 4-way strip parallelism** (the strongest
  4-core play available to an OpenCV user): big scenes still favored
  cvmatch 1.9-3.9x (button 74.5 vs 32 ms, 4K button 511 vs 133 ms,
  1080p/128 105 vs 56 ms), while two single-tile scenes flipped to strips
  (panel 86 vs 104 ms, building 24 vs 39 ms) — exactly the intra-tile
  residual documented in the headroom section. Like-for-like single-thread
  numbers are in the thread/CGO matrix below.
- **Verified-equal outputs.** `TestFullMapParityWithNativeCpp` compares every
  element of `cvmatch.MatchMap` against the response map dumped by the C++
  binary — all 13 scenes agree (see Accuracy below), so no speed comes from
  computing something different.

Reproduce locally with `cd bench && go test -bench . -benchtime 5x` and
`bench/cpp/build.sh && bench/cpp/native_bench bench/cpp/scenes 5`, or let
**GitHub Actions** do it: the [CI workflow](.github/workflows/ci.yml) re-runs
the parity tests, the Go benchmark suite, the native C++ benchmark, the
peak-RSS probe and the size report on each push to `main` (and on demand via
*workflow dispatch*), publishing everything in the job summary.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/bench-dark.svg">
  <img alt="Benchmark: cvmatch is 1.9-9.9x faster than native OpenCV C++ at identical output (4 threads, bit-identical), 10-16x in grayscale mode; cv2's Go wrapper adds only ~0-4% over native C++" src="docs/bench-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/mem-dark.svg">
  <img alt="Peak memory: 35.6 MB (4 workers) vs 145.7 MB for one 1080p match" src="docs/mem-light.svg">
</picture>

### Native code size (linux/amd64)

| artifact | size |
|---|---|
| cv2 bundled static libs (`libopencv_core.a` + `libopencv_imgproc.a` + zlib + wrapper) | 16.1 MB |
| `libcvmatch.a` | **35 KB** (~460x smaller) |
| minimal linked Go binary (`-ldflags "-s -w"`) | 5.82 MB (cv2) vs **1.57 MB** (cvmatch) |

### Accuracy

Three independent checks, all running in CI on **both cores** (the cgo and
CGO_ENABLED=0 suites each execute everything below):

1. **Element-wise parity vs native C++ OpenCV**
   (`TestFullMapParityWithNativeCpp`): the C++ binary dumps its full CV_32F
   response map (`native_bench scenes 1 dump`) and every element of
   `cvmatch.MatchMap` is compared against it — each core gets its own CI
   run. All 16 scenes agree, including a **varying-alpha** scene that
   defeats the constant-channel skip (worst diff `1.6e-06`) and two
   **low-score scenes** that pin the mid- and noise-range of the map, not
   just perfect peaks. Envelope: `~2e-06` on noise scenes,
   `4e-06`–`8e-05` on photographs, `4.7e-04` only in near-flat UI regions
   where the true value is ~0 and float32 rounding dominates.
2. **Three-way maxVal agreement on imperfect matches**
   (`TestThreeWayValues`): not every scene is a perfect crop — a
   noise-degraded template peaks at 0.904 and an absent patch at 0.180 —
   and all three implementations report the same value:

   | scene | native C++ | cvmatch cgo | cvmatch pure-Go |
   |---|---|---|---|
   | degraded_button (noisy template) | 0.903974 @(893,614) | 0.903974 @(893,614) | 0.903974 @(893,614) |
   | absent_patch (not in the image) | 0.180137 | 0.180136 | 0.180136 |
   | window button (perfect crop) | 0.999997 @(893,614) | 0.999996 @(893,614) | 0.999996 @(893,614) |

   (For the absent patch the whole map is noise-level, so the argmax
   legally lands on different near-tied bumps; the value is what matters.)
3. **Float64 brute-force reference** (main module): every response-map element
   stays within `1e-4` of a from-the-definition implementation across
   single/multi-channel, strided, and degenerate inputs (1x1 templates,
   template == image, zero-variance templates, varying alpha), and
   `TestFindAllOccurrences` verifies the MatchMap threshold+NMS recipe by
   recovering five planted copies exactly.

## Algorithm

The pipeline is exactly OpenCV's `matchTemplate(TM_CCOEFF_NORMED)`
(`modules/imgproc/src/templmatch.cpp`). Verified by diffing the C++ sources
directly: **4.8.1 → 4.12.0** changes only error-constant naming, an
expression refactor in the *masked* path (not this pipeline — an empty mask
takes the unmasked path replicated here) and an IPP enum cosmetic;
**4.12.0 → 5.x** only removes the legacy C-API `cvMatchTemplate` shim. The
unmasked TM_CCOEFF_NORMED pipeline is byte-identical across 4.8.1, 4.12.0
and 5.x:

1. `crossCorr()` — cross-correlation computed with block-based DFT
   (overlap-save tiling, `blockScale = 4.5`, `minBlockSize = 256`, template
   spectrum transformed once per channel, per-channel accumulation);
2. `common_matchTemplate()` — per-window normalization
   `num = corr - Σ wndSum_k·templMean_k`,
   `den = sqrt(max(wndSum2 - wndMean2, 0)) · templNorm`, including OpenCV's
   exact guards: the `min(0.5, 10·FLT_EPSILON·wndSum2)` rounding cutoff and
   the `1.125` saturation band;
3. `minMaxLoc()` — row-major scan, first occurrence wins on ties.

Window sums and sums-of-squares are integers accumulated in `double`, so they
are **bit-identical** to OpenCV's integral-image values; the only float32
rounding source is the correlation itself, which OpenCV also computes in
float32 — that is why parity holds to ~1e-6 on well-conditioned scenes.

### Differences from OpenCV, exhaustively

Three categories, from harmless to functional:

**A. Implementation differences with provably zero output change**

| difference | why the output is identical |
|---|---|
| sliding column sums instead of `sum`/`sqsum` integral images | both are exact integer arithmetic in `double` (< 2^53) — bit-identical window statistics |
| `minMaxLoc` fused into the normalization pass | same row-major scan, same strict-comparison first-occurrence tie rule, comparing the same rounded float32 values OpenCV would scan |
| constant-channel (alpha) skip | algebraic cancellation: a per-image-constant channel contributes exactly 0 to numerator, denominator and `templNorm` (dedicated test asserts cn=3 ≡ cn=4) |
| zero-copy input, template spectrum pre-scaled by `1/(dftW·dftH)` | pure plumbing; same numbers enter the math |

**B. Float32 rounding-path differences (bounded, measured)**

| difference | consequence |
|---|---|
| power-of-two FFT (radix-2, two-real-rows-per-complex-FFT) instead of OpenCV's mixed-radix 2^a·3^b·5^c DFT | different DFT sizes and summation orders ⇒ different float32 rounding, *not* different math. Measured element-wise vs the C++ binary: ≤ 2.4e-06 on noise, ≤ 8.1e-05 on photos, ≤ 4.7e-04 only in near-flat UI regions where the true value is ~0 |
| DFT scratch capped at 2^21 complex elements (OpenCV has no cap) | for pathologically large templates the tiling gets finer — again a rounding-path change only |
| compiler FMA contraction in the C core (gcc `-O3`) vs none in OpenCV's own kernels | sub-ulp differences folded into the same ≤1e-6 envelope |

**C. Scope differences (things cvmatch deliberately does not do)**

- Only `TM_CCOEFF_NORMED` — the one method `cv2.Match` uses. The other five
  methods (SQDIFF/CCORR families) are not implemented.
- No mask support (`cv2.Match` passes an empty mask, so OpenCV takes the
  unmasked path this library replicates).
- Input is 8-bit, 1–4 channels; OpenCV also accepts CV_32F input.
- Errors surface as Go panics instead of C++ exceptions (matching cv2's
  observable behaviour).

Nothing in category A or B changes which location wins or by how much beyond
float32 noise — that is what the four parity test layers pin down on every
run.

### The full matrix: CGO on/off x threads (measured)

Every cell measured in one session (`-cpu 1,4`, `-benchtime 5x`; CI reruns
this matrix on every push). `Match`, milliseconds:

| scene | cgo 1T | cgo 4T | pure-Go 1T | pure-Go 4T | cv2 (1T, context) |
|---|---|---|---|---|---|
| window 1600 button 96×32 | 92.8 | **30.3** | 383.8 | 129.6 | 241 |
| window 1600 icon 24×24 | 76.9 | **22.7** | 324.4 | 88.3 | 227 |
| window 1600 panel 300×200 | 107.2 | 104.3 | 499.7 | 482.7 | 251 |
| window 4K button 96×32 | 483.5 | **134.4** | 1908.9 | 527.2 | 1212 |
| noise 720p sub 96 | 76.9 | **43.7** | 373.6 | 221.7 | 149 |
| noise 1080p sub 128 | 137.4 | **55.5** | 674.2 | 249.7 | 393 |
| noise 1080p sub 32 | 104.2 | **29.6** | 416.3 | 117.1 | 299 |
| noise 4K sub 256 | 721.1 | **270.0** | 3308.1 | 1184.9 | 1931 |
| photo fruits (single tile) | 12.8 | 11.4 | 60.2 | 52.7 | 51 |
| photo building (single tile) | 40.9 | 39.3 | 217.5 | 208.5 | 93 |

Selected other rows (1080p/128): `MatchGray` 58.8 / 24.7 / 235.0 / 93.7 ms.

Readings: threading scales 2.5-3.8x wherever the planner produces multiple
tiles and ~1x on single-tile scenes (only normalization bands parallelize —
re-tiling would change the FFT rounding path; that residual is headroom
item 1). The pure-Go core is 4-5x slower than the C core (gc emits scalar
code vs 8-wide AVX2) yet still beats cv2-with-OpenCV in grayscale on every
scene and in RGBA on the multi-tile ones — with zero toolchain and zero
native bytes. `TestPureGoMatchesCgo` and `TestThreadsBitIdentical*` pin all
of these variants to the same output.

## Why it can beat OpenCV at identical output

OpenCV's `matchTemplate` is not slow because of sloppy code — it is slow
because of **what its generic pipeline does per call**. Count the work for
one 1920×1080 RGBA `Match` (template 128×128):

| per-call work | OpenCV (cv2 path) | cvmatch |
|---|---|---|
| input handover | `image.Image` → redraw/copy into `Mat` (8.3 MB) | zero-copy: C reads Go's `*image.RGBA` buffer in place |
| channels correlated | 4 (alpha included) | 3 (constant alpha provably contributes zero — skipped) |
| DFT row transforms | complex path per channel | 2 real rows packed per complex FFT (half the row FFTs), zero pad rows skipped |
| DFT column transforms | strided per-column walks | batched butterflies over contiguous rows (vectorizes, cache-friendly) |
| window statistics | materialize `sum` + `sqsum` double integral images: **~132 MB written to DRAM, then read back** | O(width) sliding column sums: **~60 KB working set, stays in L1/L2** |
| min/max | separate `minMaxLoc` pass over the 8 MB result | fused into the normalization pass |
| result values | — | element-wise equal (verified vs the C++ binary) |

The decisive line is the integral images: on a modern CPU a 1080p RGBA
`matchTemplate` is largely **memory-bandwidth-bound**, and ~140 MB of DRAM
traffic per call is pure overhead when the same integer window sums can be
produced incrementally in cache. The identities that make the shortcuts safe
are exact, not approximate:

- a sliding column sum over integers in `double` yields *bit-identical*
  values to subtracting integral-image corners — both are exact integer
  arithmetic below 2^53;
- for a channel that is constant within each image, `templSdv = 0` and
  `corr − wndSum·templMean` cancels algebraically, so dropping the alpha
  plane changes no output bit;
- the correlation itself stays float32 block-DFT — the same numeric path
  OpenCV uses — which is why the response maps agree to ~1e-6.

Three hypotheses the benchmarks rule out, all measured on the same scenes:

1. **Not the Go wrapper.** Native C++ linked against the identical static
   libs is within ~0-4% of cv2's Go API on every scene.
2. **Not the distro/build flags.** Ubuntu 24.04's own OpenCV 4.6 build lands
   within ±7% of the bundled build.
3. **Not the missing IPP.** `ipp_matchTemplate` bails out on any
   multi-channel input and never covers `TM_CCOEFF_NORMED`, so IPP cannot
   accelerate this call — and empirically a `WITH_IPP=ON` build of the same
   4.12 source (official defaults, IPPICV 2022.1.0, Release, full AVX2/AVX512
   dispatch) measured **1.2-3.7x slower** on these scenes (e.g. 725 ms vs
   245 ms on window/button, 1109 ms vs 293 ms on noise-1080p/32) because the
   IPP DFT path inside `crossCorr` underperforms OpenCV's own DFT here.
   cv2's `WITH_IPP=OFF` build choice is a win, not a handicap.

## Concurrency: what callers get for free vs what needs the library

Every public function is **safe for concurrent use** — there is no shared
mutable state; each call allocates its own scratch (`TestConcurrentMatch`
runs the suite under `-race`). That splits parallelism into two distinct
problems:

- **Throughput (many matches): just use goroutines — no library support
  needed.** Matching 100 screenshots, or one screenshot against 20 button
  templates, scales to ~N× on N cores by running `Match` calls concurrently.
  This is the recommended pattern and it already works.
- **Latency (one big match): the library now does this for you.** One call
  spreads its FFT tiles and normalization bands across GOMAXPROCS workers
  (capped at 16) with **bit-identical output for any worker count**
  (`TestThreadsBitIdentical` asserts byte-equal maps for 1/2/3/4/8 workers
  in both cores): 4K button 503→133 ms, 1080p/128 140→56 ms on 4 cores.
  Single-tile scenes (small photos) parallelize only the normalization pass
  by design — re-tiling would change the FFT rounding path.
- **DIY middle ground:** because every TM_CCOEFF_NORMED output depends only
  on its own window, a caller *can* split one large parent into horizontal
  strips overlapping by `subHeight−1` rows, `Match` them concurrently and
  merge the extrema. Results land in the same float32-rounding class as the
  whole-image call (the FFT tiling shifts, so values differ at ~1e-6 like
  any replan); ties at the strip merge need the row-major-first rule to
  mirror `minMaxLoc` exactly.

## Recipe: describing a window (one frame, many ROIs, many templates)

The building blocks already compose — no dedicated API needed. Convert the
frame to gray **once**, take zero-copy `SubImage` views per ROI, and each ROI
may check any number of expected templates (pre-convert templates to gray
once at startup; they are reused every frame):

```go
// once per frame (~7 ms at 1080p when the screenshot is RGBA;
// free if your source is already *image.Gray or JPEG/YCbCr)
gray := image.NewGray(frame.Bounds())
draw.Draw(gray, gray.Bounds(), frame, frame.Bounds().Min, draw.Src)

type expect struct {
	name string
	tpl  image.Image // pre-converted gray, reused across frames
}
groups := map[image.Rectangle][]expect{
	okArea:   {{"ok", okTpl}, {"ok_disabled", okDisabledTpl}},
	trayArea: {{"wifi", wifiTpl}, {"muted", mutedTpl}},
}

for roi, expects := range groups {
	view := gray.SubImage(roi) // zero-copy view; search cost ~ ROI area
	for _, e := range expects {
		_, _, _, score, x, y := cvmatch.MatchGray(view, e.tpl)
		if score >= 0.95 {
			fmt.Printf("%s at (%d,%d) score=%.3f\n",
				e.name, roi.Min.X+x, roi.Min.Y+y, score) // frame coords
		}
	}
}
```

Notes: returned coordinates are ROI-relative — add `roi.Min`. Every function
is safe for concurrent use, so wrapping the outer loop in goroutines is fine
(one goroutine per ROI is a good shape). Small-ROI searches cost well under
a millisecond each; the per-frame gray conversion is usually the largest
single line item, which is why it is hoisted out of the loop.

## Is OpenCV's algorithm the optimum? Remaining headroom

Deep-dive, with every claim either measured on this codebase or adversarially
reviewed. First, where a 1080p/128 `Match` actually spends its time (C core,
stage-isolated harness):

| stage | gray (1ch) | RGBA (3ch after alpha skip) | share |
|---|---|---|---|
| plan construction | ~0 ms | ~0 ms | ~0% |
| block-FFT correlation | 40.5 ms | 122.7 ms | **80–87%** |
| normalization + fused minMaxLoc | 10.8 ms | 17.4 ms | 13–20% |

**Verdict on the algorithm itself: right race, not the fastest car.** The
block-FFT + window-statistics skeleton is the correct asymptotic choice —
overlap-save tiling makes it O(N·log M) in template size M, and no exact
dense-correlation algorithm asymptotically better than FFT-based is known
(Winograd trades multiplies for more additions; a proof of optimality does
not exist either — unrestricted lower bounds remain open). But "right
asymptotic class" is where OpenCV's optimality ends: its implementation of
that skeleton loses 2–3.8x to this one at identical outputs (integral
images, 4 forced channels, unfused scans), and the list below is headroom
*this* implementation still leaves on the table under the same
results-identical-to-OpenCV contract.

Ranked remaining levers (all preserve outputs; first two are the big ones):

1. **Multithreading — now implemented (measured 2.5–3.8x on 4 cores for
   multi-tile scenes).** Correlation tiles stride across workers (each with
   private scratch, channels accumulated in order per tile) and
   normalization splits into row bands, each rebuilding its own column sums
   — bit-exact *only because* the sums are integer-valued doubles (< 2^53,
   every add exact, hence order-independent); the same trick on float inputs
   would not be. Extrema merge band-in-order to keep OpenCV's
   first-occurrence tie rule. `TestThreadsBitIdentical` asserts byte-equal
   maps for any worker count. The known residual: single-tile scenes (small
   photos) only parallelize normalization — the finer bit-exact axes
   (independent row-FFT pairs, width-chunked column butterflies) are still
   on the table for them.
2. **A better FFT kernel.** The current one is a clean radix-2 with full
   4-mul complex twiddles. Split-radix cuts operation counts ~31% at these
   sizes (radix-4 only ~15%; special-casing trivial twiddles is a few
   percent for free). Wall-clock gains will be smaller than op counts — FMA
   units absorb multiply savings and the column pass edges toward bandwidth —
   but this attacks the 80–87% slice, alongside hand-written SIMD instead of
   compiler autovectorization.
3. **Mixed-radix (2·3·5) DFT sizes** would cut power-of-two padding waste (up
   to 2x per dimension worst case; exactly zero in lucky cases — 897+127
   lands on 1024 precisely).
4. **Output-pruned inverse column FFT** (ancestor pruning of the existing
   butterfly network — the variant that keeps retained outputs bit-identical;
   Sorensen-Burrus decomposition would not). Only the first `blockH` of
   `dftH` output rows are consumed. O(n·log M) is real, but against this
   planner's actual regimes it is worth ~1.2–1.4x of one FFT pass in capped
   configurations and ≤~15% end-to-end — a niche win.
5. **Normalization: break the recurrence, don't chase the sqrt.** An ablation
   measurement refuted the obvious guess: deleting the *entire*
   sqrt+divide+guard cluster speeds the pass only 1.17–1.34x. The pass is
   latency-bound on the loop-carried double-add chains of the sliding sums
   (~20 cycles/pixel), not on sqrt and not on DRAM (traffic is within
   ~1.1–1.3x of the required floor). The bit-exact fix is reassociating the
   integer-valued sums (prefix sums / independent SIMD lanes) — legal for
   the same exactness reason as lever 1 — which then unlocks SIMD sqrt/div.
   Ceiling: the whole pass is 11–18 ms at 1080p.
6. **Plumbing:** plan/twiddle/buffer caching across calls, and skipping the
   normalized write-back when the caller wants only `Match` extrema — a few
   ms each.

**What the contract forbids — including being *better*.** With 8-bit inputs,
a number-theoretic transform over the prime 29·2^57+1 computes the raw
correlation *exactly* in integers (one prime covers any realistic template;
the two-rows-per-transform packing even survives, since −1 has a square root
mod p) at the same O(N log N) — strictly more accurate than OpenCV's float32
DFT wherever float32 rounds, at a 2–5x per-op cost on current hardware.
Requiring results identical to OpenCV rules it out: the constraint forbids
not only approximations (pyramids, early-termination bounds, feature
matching) but also exactness OpenCV itself doesn't have. That is the
cleanest evidence that OpenCV's pipeline is an engineering compromise, not
an optimum: correct asymptotics, beatable constants, and sub-optimal
accuracy by design.

Where exactly would NTT output differ from OpenCV? Only in the raw
correlation's rounding. For a 128×128 RGB template, `corr` reaches ~1e9
while float32 resolves only ~64 apart at that magnitude, so OpenCV's (and
cvmatch's) correlation is off by up to ±32 before normalization — that is
precisely the ~1e-6 noise the parity tests measure, and NTT removes it.
This mode was **built, measured, and then removed** (`MatchExact`, present
only in the v1.1.x tags): the Montgomery-NTT correlation worked and was
bit-identical across platforms and both cores, but measured ~17x slower
than `Match` (64-bit modular butterflies have no SIMD; the two-real-rows
packing does not survive in Z_p, doubling transform count and column
width — a data point that also refutes the intuition "integer arithmetic
is faster than floating point" for this workload). Since this library's
product goal is speed at OpenCV-identical output, exceeding OpenCV's
accuracy at a 17x cost earned no keep; the analysis stays recorded here.

## Optimization details

Same math, different engineering. Each item below is a measured win over the
OpenCV pipeline that cv2/gocv ship:

- **No integral images.** OpenCV materializes full double-precision `sum` and
  `sqsum` integrals before normalizing — for a 1080p RGBA parent that is
  ~132 MB of writes before any matching happens. cvmatch produces the same
  window statistics from **O(width) sliding column sums** (one `double` per
  column per channel, updated by one add/subtract per row step). That single
  change is most of the 10x peak-memory gap and removes an entire memory-bound
  pass.
- **Fused normalize + minMaxLoc.** The normalization pass tracks min/max and
  their first-occurrence locations while it walks the correlation buffer
  (identical tie semantics to OpenCV's scan), so `Match` never needs a second
  pass or a second buffer.
- **2-for-1 real FFT.** The DFT is a compact power-of-two radix-2 kernel. Row
  transforms pack **two real rows into one complex FFT** (re/im) and untangle
  the two spectra afterwards — halving row-FFT work exactly where OpenCV uses
  its real-DFT machinery. Fully-zero padding row pairs are skipped.
- **Column FFTs run batched.** The column-direction butterflies iterate over
  contiguous row segments (`spec[row][0..hw)`), so the hot inner loop is a
  straight-line vectorizable sweep instead of a strided walk — cache-friendly
  and auto-vectorized.
- **Runtime AVX2/FMA dispatch.** The hot functions are compiled twice via
  `__attribute__((target_clones("arch=haswell","default")))`; glibc's ifunc
  resolver picks the AVX2+FMA clones at load time. Portable baseline
  everywhere else (`-DCVM_NO_TARGET_CLONES` opts out), and the whole archive
  still fits in 30 KB.
- **Template spectrum pre-scaled.** The `1/(dftW·dftH)` inverse-DFT factor is
  folded into the template spectrum once per channel, so the per-block inverse
  transform needs no extra normalization sweep.
- **Provably skippable constant channels.** For a channel that is constant
  within each image, `templSdv = 0`, the window variance contribution is 0 and
  `corr - wndSum·templMean` cancels exactly — so it contributes *nothing* to
  CCOEFF_NORMED. `Match` detects a constant alpha plane (virtually every
  screenshot) with a cheap scan and processes 3 of 4 channels: ~25% less work,
  bit-equivalent output (there is a dedicated test asserting cn=3 == cn=4).
- **Zero-copy inputs.** `*image.RGBA`, `*image.Gray` and the Y plane of
  `*image.YCbCr` (JPEG) are handed to C with their native strides — sub-images
  included, no redraw, no `Mat`, no copies. The uint8→float conversion happens
  inside the block-load loop, which is a copy the FFT needs anyway; the parent
  image is never converted or duplicated as a whole.
- **Bounded scratch.** DFT tile buffers are capped (~16 MB each even for
  pathological template sizes; a few hundred KB in typical scenarios), and the
  only full-size allocation is the float32 response map itself. Everything is
  freed before returning — one 1080p `Match` peaks at ~15 MB of native memory
  and performs a single 32-byte Go allocation.
- **`MatchGray`** trades exact RGBA parity for a single-channel pipeline
  (OpenCV's fixed-point BT.601 RGB2GRAY weights; YCbCr images use their Y
  plane directly with zero conversion): ~4x less FFT and normalization work,
  which is the right default when hunting UI elements in screenshots.

## Layout

- `cvmatch.c` / `cvmatch.h` — the native C core (C99, no deps).
- `impl_purego.go` — the pure-Go port of the same core; `impl_cgo.go` /
  `impl_nocgo.go` select between them by build tag.
- `cvmatch.go` — public API and zero-copy image conversion.
- `scenes/` — deterministic benchmark scenes shared by the main module's
  cgo-vs-pure-Go benchmarks and the cv2 comparison in `bench/`.
- `bench/` — separate module holding the native-C++ comparison
  (element-wise parity test, `memprobe` peak-RSS tool, binary-size probe)
  so the root module stays dependency-free. Through v1.2.x it also carried
  the `hkloudou/cv2` Go-wrapper comparison, retired after settling that the
  wrapper adds only ~0-4% over native C++.
- `bench/cpp/` — native OpenCV C++ benchmark linked against the same static
  libraries that cv2 bundles (`build.sh` fetches matching headers;
  `cmd/dumpscenes` exports byte-identical scene images).
- `bench/testdata/fetch.sh` — downloads the real sample photographs.
- `bench/cmd/annotate` — renders the match-result demo images shown above
  (published on the `assets` branch).
- `docs/genchart.py` — regenerates the README charts from benchmark numbers.
- `make lib` — builds the standalone `libcvmatch.a` for non-Go consumers.

## Requirements

None beyond Go. With cgo enabled (default) any C99 compiler gives the fast
C core — Linux, macOS, Windows (mingw); AVX2 multi-versioning engages
automatically on x86-64 glibc and falls back to portable code elsewhere.
With `CGO_ENABLED=0` (or no C toolchain at all, e.g. cross-compiling) the
pure-Go core is selected automatically — same results, no setup.
