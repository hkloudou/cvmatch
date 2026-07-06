# cvmatch

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml) [![Go Reference](https://pkg.go.dev/badge/github.com/hkloudou/cvmatch.svg)](https://pkg.go.dev/github.com/hkloudou/cvmatch) ![CGO optional](https://img.shields.io/badge/CGO-optional-success)

OpenCV-compatible `TM_CCOEFF_NORMED` template matching for Go, with no
OpenCV, no bundled static libraries and no dependencies. Two interchangeable
cores ship in the package:

- **cgo core** (default): one dependency-free C99 file, ~35 KB of native
  code including threading and AVX2 multi-versioning; compiles in ~1 s
  during `go build`.
- **pure-Go core** (`CGO_ENABLED=0`, or no C toolchain): a port of the same
  algorithm, selected automatically, so plain cross-compilation works.

`cvmatch.Impl` reports which core is active (`"cgo"` / `"purego"`). The two
cores, and native OpenCV C++ itself, are pinned to the same output by the
test suite — see [Accuracy](#accuracy-cgo--pure-go--native-c-compared)
for the measured three-way comparison, and
[Benchmarks](#benchmarks-cgo--pure-go--native-c) for where each one is
faster and where it is not.

```go
import "github.com/hkloudou/cvmatch"

// matchTemplate(TM_CCOEFF_NORMED) + minMaxLoc in one call:
minVal, minX, minY, maxVal, maxX, maxY := cvmatch.Match(parent, sub)

// Grayscale fast path (~2.5x faster again; ideal for screenshots/UI
// hunting). Alpha is dropped here by design — the same semantics as
// OpenCV's COLOR_RGBA2GRAY; use Match when alpha carries signal:
minVal, minX, minY, maxVal, maxX, maxY = cvmatch.MatchGray(parent, sub)

// Full response map (find every occurrence above a threshold):
resp, w, h := cvmatch.MatchMap(parent, sub)
```

`Match` panics if an image is empty or `sub` is larger than `parent`
(mirroring OpenCV's assertion on the same condition).

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
kept for six-tuple compatibility with OpenCV pipelines, where the SQDIFF
methods treat the *minimum* as best.

`MatchMap` returns the whole score map (`resp[y*w+x]` = score of placing the
template's top-left corner at `(x, y)`), which unlocks what a single best
hit cannot: finding **every** occurrence (threshold + local-maximum
suppression with a radius about half the template size — verified by
`TestFindAllOccurrences`, which recovers five planted copies exactly),
**ambiguity checks** (if the second-highest peak nearly ties the highest —
three look-alike buttons — the match is not trustworthy even at
`maxVal≈1`), ROI-restricted search, sub-pixel refinement by fitting the
peak's neighborhood, and heatmap debugging.

Below, the green box is drawn at the location `Match` returned for each
template (images live on the [`assets`](../../tree/assets) branch; rendered
by `bench/cmd/annotate`):

| template | found in parent | result |
|---|---|---|
| ![button](../../raw/assets/demo/window1600_button96x32.tpl.png) | ![window](../../raw/assets/demo/window1600_button96x32.jpg) | `maxVal=0.999996` @ (893,614) — the right button among three look-alikes |
| ![fruits template](../../raw/assets/demo/photo_fruits.tpl.png) | ![fruits](../../raw/assets/demo/photo_fruits.jpg) | `maxVal=1.000000` @ (305,210) |
| ![baboon template](../../raw/assets/demo/photo_baboon.tpl.png) | ![baboon](../../raw/assets/demo/photo_baboon.jpg) | `maxVal=1.000000` @ (250,180) |
| ![building template](../../raw/assets/demo/photo_building.tpl.png) | ![building](../../raw/assets/demo/photo_building.jpg) | `maxVal=1.000000` @ (420,240) |
| ![graf template](../../raw/assets/demo/photo_graf1.tpl.png) | ![graf](../../raw/assets/demo/photo_graf1.jpg) | `maxVal=0.999999` @ (350,260) |
| ![starry template](../../raw/assets/demo/photo_starry_night.tpl.png) | ![starry night](../../raw/assets/demo/photo_starry_night.jpg) | `maxVal=0.999999` @ (400,300) |
| ![alpha template](../../raw/assets/demo/noise640_alpha.tpl.png) | ![varying alpha](../../raw/assets/demo/noise640_alpha.png) | `maxVal=1.000000` @ (217,143) — **varying-alpha** scene; the PNG stores the tested bytes verbatim (straight alpha, per-pixel 64–255), so it renders genuinely semi-transparent |

## Alpha and semi-transparency: what OpenCV actually does

Questions about the alpha channel come up often enough to answer precisely.
All claims below were checked against OpenCV's C++ source
(`modules/imgproc/src/templmatch.cpp`), not its documentation.

- **`matchTemplate` has no transparency semantics.** On CV_8UC4 input the
  alpha plane is a plain fourth data channel: the source contains no alpha
  handling anywhere — every channel loop is unconditional, so alpha bytes
  enter the sums and the correlation exactly like color bytes. Nothing is
  composited, blended or weighted. cvmatch replicates that: a
  semi-transparent pixel is just four numbers.
- **By default OpenCV never even sees alpha.** `cv::imread` strips the
  alpha channel unless explicitly asked to keep it (`IMREAD_UNCHANGED`),
  so a typical OpenCV pipeline matches 3-channel data.
- **"Transparency-aware" matching is a different OpenCV feature.**
  `matchTemplate` accepts an optional mask argument, which weights template
  pixels (a mask can be derived from the template's alpha plane). That is a
  separate, substantially slower code path; the classic
  4-channel-image-to-Mat pipeline passes no mask, and cvmatch implements
  only the unmasked path (a documented scope difference — see below).
- **The constant-alpha skip changes nothing.** Screenshots almost always
  carry alpha = 255 everywhere. For a channel that is constant within each
  image, its contribution to the CCOEFF_NORMED numerator, denominator and
  template norm cancels *algebraically*, so `Match` skips it — ~25% less
  work, provably identical output. The skip triggers only when **both**
  images have a constant alpha plane; tests cover the skip
  (`TestConstantAlphaSkip`, four constants including 0/255, asserted on
  **both cores**), the non-skip (`TestVaryingAlpha`,
  `TestAlphaMixedConstancy`), and the fact that alpha genuinely matters
  when it varies (`TestAlphaMatters`: cn=3 vs cn=4 maps differ materially
  — also on both cores). The varying-alpha scene is additionally pinned
  element-wise against native C++ output in CI.
- **A Go-specific gotcha, documented for fairness:** Go's `*image.RGBA` is
  alpha-**premultiplied** by convention, while PNG (and `image.NRGBA`, and
  OpenCV's `IMREAD_UNCHANGED`) store **straight** alpha. cvmatch consumes
  `*image.RGBA` bytes as-is and redraws other formats through
  `image/draw` (which premultiplies) — identical to the classic
  ImageToMatRGBA pipeline it replaces. Feeding the same bytes to cvmatch
  and OpenCV gives the same map (that is exactly what the parity harness
  does); but decoding a semi-transparent PNG in Go and in
  `imread(IMREAD_UNCHANGED)` produces *different byte values* for the
  non-opaque pixels before any matching starts. This also bit the demo
  image above: `png.Encode` un-premultiplies `*image.RGBA` pixels on
  encode, which wraps values in test data whose R/G/B exceed A — the demo
  is therefore encoded from the identical bytes reinterpreted as NRGBA, so
  the published file byte-for-byte equals the tested input.

## Accuracy: CGO / pure-Go / native C++ compared

The claim "same result as OpenCV" is measured, not asserted, and it is
measured on **all three implementations** over the full scene set — not
only on perfect crops. Two scenes are deliberately degraded so the peak
lands mid-range, and one searches a patch that is not in the image at all,
so the noise floor of the map is compared too.

**`Match` (best value), all three implementations** — from
`TestThreeWayValues`, run in CI for each core against the same response
maps dumped by the C++ binary; the four rows below span the score range
(the full 17-scene log is printed in every CI run):

| scene | peak | native C++ | cvmatch cgo | cvmatch pure-Go |
|---|---|---|---|---|
| window button (perfect crop) | ~1.0 | 0.999997 @(893,614) | 0.999996 @(893,614) | 0.999996 @(893,614) |
| degraded_button (noisy template) | high | 0.903974 @(893,614) | 0.903974 @(893,614) | 0.903974 @(893,614) |
| half_degraded_noise | mid | 0.565384 @(431,285) | 0.565385 @(431,285) | 0.565385 @(431,285) |
| absent_patch (not in the image) | noise | 0.180137 | 0.180136 | 0.180136 |

For the absent patch the whole map is noise-level, so the argmax legally
lands on different near-tied bumps; the value is what matters. Positions
agree exactly on every scene with a real peak.

**`MatchMap` (every element), each core vs native C++** — from
`TestFullMapParityWithNativeCpp`: the C++ binary dumps its full CV_32F
response map per scene and every element of `cvmatch.MatchMap` is compared
against it, once per core:

| scene group | worst element diff, cgo vs C++ | pure-Go vs C++ |
|---|---|---|
| dense noise (720p / 1080p×2 / 4K / alpha / half-degraded) | ≤ 2.4e-06 | ≤ 2.3e-06 |
| photographs (5 scenes) | ≤ 8.1e-05 | ≤ 7.0e-05 |
| synthetic UI windows (4 scenes) | ≤ 4.7e-04 | ≤ 4.7e-04 |
| degraded_button / absent_patch | ≤ 4.3e-04 | ≤ 3.6e-04 |

The larger UI-scene numbers are where the true score is ~0 in near-flat
regions and float32 rounding dominates; the *peak* values agree to 1e-6
everywhere (table above). The two Go cores are additionally compared
against **each other** element-wise (`TestPureGoMatchesCgo`, ≤1e-5 over
seven shapes including 1/3/4-channel, strided and degenerate cases), and
each core is compared against a **float64 brute-force reference**
implementing OpenCV's formulas from the definition (≤1e-4; includes 1x1
templates, template == image, zero-variance templates, varying alpha).

**Bit-identity guarantees** (exact equality, not tolerances):
`TestThreadsBitIdentical` / `TestThreadsBitIdenticalPureGo` /
`TestThreadsVar` assert byte-equal maps and extrema for 1/2/3/4/8/16
workers in both cores, and `TestConstantAlphaSkip` asserts the
constant-alpha skip is a no-op on both cores.

Everything in this section runs in CI on every push, twice: once with
`CGO_ENABLED=1` and once with `CGO_ENABLED=0`.

## Benchmarks: CGO / pure-Go / native C++

Scenarios cover a realistic workload — finding a button, a toolbar icon or
a panel inside a rendered desktop window (flat regions, gradients, text,
look-alike widgets) — plus dense-noise worst cases and **real
photographs** from
[OpenCV's `samples/data`](https://github.com/opencv/opencv/tree/4.12.0/samples/data)
(Apache-2.0; `bench/testdata/fetch.sh` downloads them, and they are
mirrored on the [`assets`](../../tree/assets) branch for viewing):

| image | size | template | content |
|---|---|---|---|
| [`fruits.jpg`](../../raw/assets/samples/fruits.jpg) | 512×480 | 80×80 @ (305,210) | fruit bowl, saturated colors |
| [`baboon.jpg`](../../raw/assets/samples/baboon.jpg) | 512×512 | 64×64 @ (250,180) | fur, high-frequency texture |
| [`building.jpg`](../../raw/assets/samples/building.jpg) | 868×600 | 100×100 @ (420,240) | architecture, repeating windows |
| [`graf1.png`](../../raw/assets/samples/graf1.png) | 800×640 | 120×120 @ (350,260) | graffiti wall (VGG dataset) |
| [`starry_night.jpg`](../../raw/assets/samples/starry_night.jpg) | 752×600 | 128×128 @ (400,300) | painting, swirling gradients |

The baseline is **native OpenCV C++**: `bench/cpp/native_bench`, plain C++
linked against prebuilt static OpenCV 4.12 archives (Release, WITH_IPP=OFF,
SIMD dispatch on), timed end-to-end per call, best-of-7. No wrapper of any
kind sits between the timer and OpenCV.

### Fairness rules

- **End-to-end timing everywhere.** Every implementation is timed over the
  complete per-call path starting from an in-memory RGBA frame — image/Mat
  conversion, matching, min/max scan, cleanup. No implementation gets to
  keep pre-converted inputs or reuse buffers across calls.
- **Byte-identical inputs.** `bench/cmd/dumpscenes` exports the exact pixel
  buffers the Go benchmarks use; the C++ benchmark consumes those dumps.
- **Representative OpenCV build.** The static 4.12 archives are a standard
  Release build. Two alternatives were measured to rule out a weak
  baseline: a distro build (Ubuntu 24.04, OpenCV 4.6.0) lands within ±7%
  on every scene, and a `WITH_IPP=ON` build of the same 4.12 source
  (official defaults, IPPICV 2022.1.0, full AVX2/AVX512 dispatch) measured
  **1.2–3.7x slower** than the IPP-off build on these scenes — the IPP DFT
  path inside `crossCorr` underperforms OpenCV's own DFT here, and
  `ipp_matchTemplate` itself bails out on multi-channel input and never
  covers TM_CCOEFF_NORMED anyway. The baseline used is the fastest OpenCV
  configuration measured.
- **Threading disclosed, both sides get the cores.** OpenCV's
  `matchTemplate` path does not use extra cores (measured: its times are
  flat across thread settings), so the honest multi-core OpenCV baseline
  is caller-side parallelism. That was measured too: splitting the parent
  into 4 overlapping strips and matching them concurrently with OpenCV
  makes the big scenes 3–3.5x faster (window button 74.5 ms, 4K button
  511 ms, 1080p/128 105 ms) — still slower than cvmatch's internal
  parallelism on those scenes, but **faster than cvmatch on single-tile
  scenes** (panel 86 vs 104 ms, building 24 vs 39 ms). Single-thread
  columns are in the matrix below for like-for-like reading.
- **Verified-equal outputs.** The parity suite above compares every
  response-map element of each core against the C++ binary — no speed
  comes from computing something different.
- **One machine, one session.** All numbers in this README come from the
  same 4-core Intel Xeon @ 2.10 GHz (linux/amd64, cloud VM); expect
  some run-to-run variance. CI re-measures everything on every push
  (`-cpu 1,4`), publishing raw numbers in the job summary — those runs,
  on whatever hardware GitHub provides, are the reproducible record.

Reproduce locally with `go test -bench . -benchtime 5x -cpu 1,4` (and
`CGO_ENABLED=0` for the pure-Go core), plus
`bench/cpp/build.sh && bench/cpp/native_bench bench/cpp/scenes 7`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/bench-dark.svg">
  <img alt="Benchmark: native OpenCV C++ vs cvmatch cgo and pure-Go cores at identical output" src="docs/bench-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/mem-dark.svg">
  <img alt="Peak memory for one 1080p match: native OpenCV C++ vs cvmatch" src="docs/mem-light.svg">
</picture>

### The full matrix: three implementations × threads (measured)

`Match`, milliseconds. Native C++ is single-threaded because that is how
OpenCV's `matchTemplate` runs; the Go columns come from `-cpu 1,4`:

| scene | native C++ | cgo 1T | cgo 4T | pure-Go 1T | pure-Go 4T |
|---|---|---|---|---|---|
| window 1600 button 96×32 | 259.8 | 108.4 | **33.9** | 446.0 | 141.3 |
| window 1600 icon 24×24 | 236.7 | 91.0 | **28.2** | 370.5 | 106.7 |
| window 1600 panel 300×200 | 260.6 | 129.4 | **121.0** | 580.2 | 556.5 |
| window 4K button 96×32 | 1274.5 | 567.9 | **164.6** | 2279.6 | 630.0 |
| noise 720p sub 96 | 151.8 | 90.3 | **51.0** | 432.7 | 254.9 |
| noise 1080p sub 128 | 428.1 | 160.2 | **66.3** | 757.4 | 291.6 |
| noise 1080p sub 32 | 306.3 | 122.0 | **40.2** | 493.3 | 149.3 |
| noise 4K sub 256 | 1815.0 | 816.2 | **282.5** | 3742.8 | 1362.4 |
| noise 640 alpha (4-channel) | 34.6 | 30.0 | **17.1** | 143.2 | 83.6 |
| photo fruits (single tile) | 62.7 | 14.1 | **13.2** | 67.4 | 61.3 |
| photo baboon | 38.3 | 14.6 | **12.6** | 67.3 | 62.7 |
| photo building | 94.6 | 45.9 | **42.1** | 245.6 | 239.6 |
| photo graf1 | 109.3 | 46.5 | **43.2** | 243.7 | 244.0 |
| photo starry_night | 82.8 | **46.2** | 50.7 | 242.1 | 235.2 |

`MatchGray` at 1080p/128 for scale: cgo 64.2 / 29.1 ms, pure-Go
285.8 / 107.8 ms (1T / 4T). Native C++ has no gray row in this suite — a
fair baseline would need cvtColor + 1-channel matchTemplate timed
end-to-end, which was not measured; treat MatchGray numbers as
cvmatch-internal.

**Readings, including where cvmatch does *not* win:**

- The **cgo core** is faster than native OpenCV on every scene
  single-threaded, but the margin varies a lot: 1.15x on the 4-channel
  varying-alpha scene (no constant-alpha skip there — 4 full channels) up
  to 4.4x on the smallest photo. Internal threading adds another
  2.2–3.5x on multi-tile scenes (final 6.4–8.4x on the big windows/noise
  scenes at 4 cores).
- Threading is ≈**neutral on single-tile scenes** (panel, the photos):
  only the normalization pass parallelizes there, because re-tiling the
  FFT would change the rounding path — and it can even cost a little
  (starry_night measured 46.2 ms at 1T vs 50.7 ms at 4T; within
  run-to-run variance but consistently not a win). OpenCV driven with
  caller-side strips *beats* cvmatch on some single-tile scenes (see
  fairness rules) — that residual is real and documented in the headroom
  section.
- The **pure-Go core is slower than native OpenCV single-threaded on
  every scene** — 1.1x (fruits) to 4.1x (varying alpha) slower; gc emits
  scalar code against OpenCV's SIMD. With 4 threads it pulls ahead on
  multi-tile scenes (button 141 vs 260 ms) but stays behind on
  single-tile ones (building 240 vs 95 ms). Its value is zero toolchain
  and zero native code, not raw speed — if you can use cgo, use the cgo
  core.
- The Go rows allocate almost nothing in cgo mode (32 B/op for `Match`);
  the pure-Go core allocates its scratch on the Go heap (tens of MB/op
  on big scenes), which is visible as GC pressure in allocation-sensitive
  services.

### Native code size and memory (linux/amd64)

| artifact | size |
|---|---|
| static OpenCV archives the C++ baseline links (`libopencv_core.a` + `libopencv_imgproc.a` + zlib) | 16.1 MB |
| `libcvmatch.a` | **35 KB** |
| minimal `Match` binary, `-ldflags "-s -w"` | **1.57 MB** (a comparable OpenCV-linked Go binary measured 5.82 MB before that comparison was retired) |

Peak whole-process memory for one 1920×1080 / 128×128 match (VmHWM, fresh
process per run): idle Go process 12.8 MB, `cvmatch.Match` 48.2 MB,
`cvmatch.MatchGray` 42.0 MB, native OpenCV C++ 165.8 MB. OpenCV
materializes full double-precision integral images (~132 MB written per
1080p RGBA call); cvmatch replaces them with O(width) sliding sums, so its
peak is dominated by the input frame and the float32 response map
themselves.

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
| constant-channel (alpha) skip | algebraic cancellation: a per-image-constant channel contributes exactly 0 to numerator, denominator and `templNorm` (`TestConstantAlphaSkip` asserts cn=3 ≡ cn=4 on both cores) |
| zero-copy input, template spectrum pre-scaled by `1/(dftW·dftH)` | pure plumbing; same numbers enter the math |

**B. Float32 rounding-path differences (bounded, measured)**

| difference | consequence |
|---|---|
| power-of-two FFT (radix-2, two-real-rows-per-complex-FFT) instead of OpenCV's mixed-radix 2^a·3^b·5^c DFT | different DFT sizes and summation orders ⇒ different float32 rounding, *not* different math. Measured element-wise vs the C++ binary — see the Accuracy table |
| DFT scratch capped at 2^21 complex elements (OpenCV has no cap) | for pathologically large templates the tiling gets finer — again a rounding-path change only |
| compiler FMA contraction in the C core (gcc `-O3`) vs none in OpenCV's own kernels | sub-ulp differences folded into the same envelope |

**C. Scope differences (things cvmatch deliberately does not do)**

- Only `TM_CCOEFF_NORMED`. The other five methods (SQDIFF/CCORR families)
  are not implemented.
- No mask support — the classic 4-channel pipeline passes an empty mask,
  which takes the unmasked OpenCV path this library replicates.
  Transparency-weighted matching therefore isn't available here (see the
  alpha section above).
- Input is 8-bit, 1–4 channels; OpenCV also accepts CV_32F input.
- Input conversion happens in Go's image model: `*image.RGBA` is consumed
  as-is (premultiplied bytes), other formats are redrawn through
  `image/draw`. OpenCV pipelines that `imread` a file get straight-alpha
  or alpha-stripped bytes instead — same matcher, different bytes in
  (see the alpha section).
- Errors surface as Go panics instead of C++ exceptions.

Nothing in category A or B changes which location wins or by how much beyond
float32 noise — that is what the parity layers in Accuracy pin down on every
run.

## Concurrency and the Threads switch

Every public function is **safe for concurrent use** — there is no shared
mutable state; each call allocates its own scratch (`TestConcurrentMatch`
runs the suite under `-race`). That splits parallelism into two distinct
problems:

- **Throughput (many matches): just use goroutines — no library support
  needed.** Matching 100 screenshots, or one screenshot against 20 button
  templates, scales to ~N× on N cores by running `Match` calls concurrently.
- **Latency (one big match): one call spreads its FFT tiles and
  normalization bands across workers with bit-identical output for any
  worker count** (asserted byte-for-byte by `TestThreadsBitIdentical*`).
  Single-tile scenes parallelize only the normalization pass by design —
  re-tiling would change the FFT rounding path.
- **DIY middle ground:** because every TM_CCOEFF_NORMED output depends only
  on its own window, a caller *can* split one large parent into horizontal
  strips overlapping by `subHeight−1` rows, `Match` them concurrently and
  merge the extrema. Results land in the same float32-rounding class as the
  whole-image call (the FFT tiling shifts, so values differ at ~1e-6 like
  any replan); ties at the strip merge need the row-major-first rule to
  mirror `minMaxLoc` exactly.

The worker count is controllable:

```go
cvmatch.Threads = 1 // pin single-threaded; 0 = automatic (GOMAXPROCS, ≤16)
```

or `CVMATCH_THREADS=1` in the environment (read once at package init).
Any value produces bit-identical results — it is a performance knob only,
mainly useful for benchmarking and for capping the library's CPU use inside
a loaded service. Set it before concurrent use; it is a plain package
variable. (The C core additionally honors a `CVM_MAX_THREADS` env cap, and
`-DCVM_NO_TARGET_CLONES` at build time disables AVX2 multi-versioning.)

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

## Why the same math runs faster here

OpenCV's `matchTemplate` is not slow because of sloppy code — the cost is in
**what its generic pipeline does per call**. For one 1920×1080 RGBA `Match`
(template 128×128):

| per-call work | OpenCV | cvmatch |
|---|---|---|
| input handover | copy into a `Mat` (8.3 MB) | zero-copy: the core reads Go's `*image.RGBA` buffer in place |
| channels correlated | 4 (alpha included) | 3 when alpha is constant (provably zero contribution) |
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

Three alternative explanations were measured and ruled out:

1. **Not binding overhead.** The baseline is plain C++ calling OpenCV
   directly. (A Go binding driving the same archives was also measured
   during development and stayed within ~0–4% of the C++ times.)
2. **Not the build flags.** A distro OpenCV (Ubuntu 24.04, 4.6.0) lands
   within ±7% of the static 4.12 build used.
3. **Not missing IPP.** `WITH_IPP=ON` measured 1.2–3.7x *slower* here (see
   fairness rules); IPP cannot accelerate this call anyway.

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
that skeleton loses measurably to this one at identical outputs (integral
images, 4 forced channels, unfused scans), and the list below is headroom
*this* implementation still leaves on the table under the same
results-identical-to-OpenCV contract.

Ranked remaining levers (all preserve outputs; first two are the big ones):

1. **Multithreading — implemented (measured 2.5–3.8x on 4 cores for
   multi-tile scenes).** Correlation tiles stride across workers (each with
   private scratch, channels accumulated in order per tile) and
   normalization splits into row bands, each rebuilding its own column sums
   — bit-exact *only because* the sums are integer-valued doubles (< 2^53,
   every add exact, hence order-independent); the same trick on float inputs
   would not be. Extrema merge band-in-order to keep OpenCV's
   first-occurrence tie rule. The known residual: single-tile scenes (small
   photos, the panel) only parallelize normalization — the finer bit-exact
   axes (independent row-FFT pairs, width-chunked column butterflies) are
   still on the table, and until they land, strip-parallel OpenCV beats
   cvmatch on some single-tile scenes (measured; see fairness rules).
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
DFT wherever float32 rounds. Requiring results identical to OpenCV rules it
out: the constraint forbids not only approximations (pyramids,
early-termination bounds, feature matching) but also exactness OpenCV itself
doesn't have. That is the cleanest evidence that OpenCV's pipeline is an
engineering compromise, not an optimum: correct asymptotics, beatable
constants, and sub-optimal accuracy by design.

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

Same math, different engineering:

- **No integral images.** OpenCV materializes full double-precision `sum` and
  `sqsum` integrals before normalizing — for a 1080p RGBA parent that is
  ~132 MB of writes before any matching happens. cvmatch produces the same
  window statistics from **O(width) sliding column sums** (one `double` per
  column per channel, updated by one add/subtract per row step). That single
  change is most of the peak-memory gap and removes an entire memory-bound
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
  everywhere else (`-DCVM_NO_TARGET_CLONES` opts out).
- **Template spectrum pre-scaled.** The `1/(dftW·dftH)` inverse-DFT factor is
  folded into the template spectrum once per channel, so the per-block inverse
  transform needs no extra normalization sweep.
- **Provably skippable constant channels.** For a channel that is constant
  within each image, `templSdv = 0`, the window variance contribution is 0 and
  `corr - wndSum·templMean` cancels exactly — so it contributes *nothing* to
  CCOEFF_NORMED. `Match` detects a constant alpha plane (virtually every
  screenshot) with a cheap scan and processes 3 of 4 channels: ~25% less work,
  bit-equivalent output, asserted on both cores.
- **Zero-copy inputs.** `*image.RGBA`, `*image.Gray` and the Y plane of
  `*image.YCbCr` (JPEG) are handed to the core with their native strides —
  sub-images included, no redraw, no copies. The uint8→float conversion
  happens inside the block-load loop, which is a copy the FFT needs anyway;
  the parent image is never converted or duplicated as a whole.
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
- `scenes/` — deterministic benchmark/parity scenes shared by the main
  module's benchmarks and the native comparison in `bench/`.
- `bench/` — separate module holding the native-C++ comparison
  (element-wise parity + three-way value tests, `memprobe` peak-RSS tool,
  binary-size probe) so the root module stays dependency-free.
- `bench/cpp/` — the native OpenCV C++ benchmark (`build.sh` fetches
  prebuilt static OpenCV 4.12 archives and matching headers;
  `cmd/dumpscenes` exports byte-identical scene images).
- `bench/testdata/fetch.sh` — downloads the real sample photographs.
- `bench/cmd/annotate` — renders the match-result demo images shown above
  (published on the `assets` branch).
- `docs/genchart.py` — regenerates the README charts from benchmark numbers.
- `make lib` — builds the standalone `libcvmatch.a` for non-Go consumers.

## Requirements and releases

None beyond Go. With cgo enabled (default) any C99 compiler gives the fast
C core — Linux, macOS, Windows (mingw); AVX2 multi-versioning engages
automatically on x86-64 glibc and falls back to portable code elsewhere.
With `CGO_ENABLED=0` (or no C toolchain at all, e.g. cross-compiling) the
pure-Go core is selected automatically — same results, slower (see the
matrix), zero setup.

Versions are tagged from `main` via the *Tag release* workflow
(`.github/workflows/tag.yml`); CI runs the full test + parity + benchmark
suite on every tag.
