# cvmatch

OpenCV-compatible `TM_CCOEFF_NORMED` template matching for Go, implemented as
**one dependency-free C file** behind cgo. No OpenCV, no bundled multi-megabyte
static libraries — the entire native code compiles to a ~30 KB archive
(~23 KB of `.text`, AVX2 function multi-versioning included).

```go
import "github.com/hkloudou/cvmatch"

// Drop-in replacement for cv2.Match (same signature, same numbers):
minVal, minX, minY, maxVal, maxX, maxY := cvmatch.Match(parent, sub)

// Grayscale fast path (~2.5x faster again; ideal for screenshots/UI hunting):
minVal, minX, minY, maxVal, maxX, maxY = cvmatch.MatchGray(parent, sub)

// Full response map (find every occurrence above a threshold):
resp, w, h := cvmatch.MatchMap(parent, sub)
```

`Match` panics if an image is empty or `sub` is larger than `parent`, matching
the behaviour of the OpenCV-backed original.

## Benchmarks

Scenarios cover the real workload — finding a button, a toolbar icon or a
panel inside a rendered desktop window (flat regions, gradients, text,
look-alike widgets) — plus dense-noise worst cases and **real photographs**
from OpenCV's `samples/data` (`bench/testdata/fetch.sh` downloads them; the
suite runs with synthetic scenes only when they are absent).

Four implementations are measured on every scene, all returning the same
location and value:

- **OpenCV C++ (native)** — `bench/cpp/native_bench`, plain C++ linked
  against the *exact same* prebuilt static OpenCV 4.12 libraries that
  `hkloudou/cv2` bundles, timed **end-to-end per call** for fairness (from an
  in-memory RGBA buffer: Mat copy → `matchTemplate` → `minMaxLoc` → release),
  best-of-5;
- **cv2.Match (Go)** — [`hkloudou/cv2`](https://github.com/hkloudou/cv2)
  v0.41200.0 through its Go API;
- **cvmatch.Match** / **cvmatch.MatchGray** — this library.

> **Is the Go wrapper the bottleneck? No.** Native C++ and cv2's Go API land
> within ~0-4% of each other on every scene (e.g. 1080p/128: 390.5 ms native
> vs 392.7 ms Go; window/button: 245.0 vs 241.1 ms). The cost is OpenCV's own
> pipeline — 4-channel DFT correlation plus full double-precision integral
> images — which is exactly what cvmatch restructures: **2.0-3.8x faster than
> native OpenCV C++ at identical output values, 3.9-8.4x in grayscale mode.**

Reproduce locally with `cd bench && go test -bench . -benchtime 5x` and
`bench/cpp/build.sh && bench/cpp/native_bench bench/cpp/scenes 5`, or let
**GitHub Actions** do it: the [CI workflow](.github/workflows/ci.yml) re-runs
the parity tests, the Go benchmark suite, the native C++ benchmark, the
peak-RSS probe and the size report on each push to `main` (and on demand via
*workflow dispatch*), publishing everything in the job summary.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/bench-dark.svg">
  <img alt="Benchmark: cvmatch is 2.0-3.8x faster than native OpenCV C++ at identical output, 3.9-8.4x in grayscale mode; cv2's Go wrapper adds only ~0-4% over native C++" src="docs/bench-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/mem-dark.svg">
  <img alt="Peak memory: 14.8 MB vs 145.5 MB for one 1080p match" src="docs/mem-light.svg">
</picture>

### Native code size (linux/amd64)

| artifact | size |
|---|---|
| cv2 bundled static libs (`libopencv_core.a` + `libopencv_imgproc.a` + zlib + wrapper) | 16.1 MB |
| `libcvmatch.a` | **30 KB** (~525x smaller) |
| minimal linked Go binary (`-ldflags "-s -w"`) | 5.82 MB (cv2) vs **1.57 MB** (cvmatch) |

### Accuracy

Three independent checks, all in CI:

1. **Element-wise response-map parity vs OpenCV** (`bench/`,
   `TestFullMapParityWithCv2`): every element of `MatchMap` is compared with
   OpenCV's `matchTemplate` output. Worst element difference: `2.2e-06` on
   noise scenes, `1.9e-05` on real photographs, `4.7e-04` on UI scenes with
   near-flat windows.
2. **min/max contract parity** (`TestParityWithCv2`): `minVal`/`maxVal` agree
   to ~6 decimals and locations are identical on all thirteen scenarios.
3. **Float64 brute-force reference** (main module): every response-map element
   stays within `1e-4` of a from-the-definition implementation across
   single/multi-channel, strided, and degenerate inputs (1x1 templates,
   template == image, zero-variance templates, varying alpha).

## Algorithm

The pipeline is exactly OpenCV's `matchTemplate(TM_CCOEFF_NORMED)`
(`modules/imgproc/src/templmatch.cpp`; the algorithm is identical in 4.8.1 and
5.x):

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
- **Runtime AVX2/FMA dispatch.** The six hot functions are compiled twice via
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

- `cvmatch.c` / `cvmatch.h` — the whole native implementation (C99, no deps).
- `cvmatch.go` — public API and zero-copy image conversion.
- `bench/` — separate module holding the `hkloudou/cv2` comparison (UI +
  noise + photo scenario builders, benchmarks, element-wise parity tests,
  `memprobe` peak-RSS tool, binary-size probes) so the root module stays
  dependency-free.
- `bench/cpp/` — native OpenCV C++ benchmark linked against the same static
  libraries that cv2 bundles (`build.sh` fetches matching headers;
  `cmd/dumpscenes` exports byte-identical scene images).
- `bench/testdata/fetch.sh` — downloads the real sample photographs.
- `docs/genchart.py` — regenerates the README charts from benchmark numbers.
- `make lib` — builds the standalone `libcvmatch.a` for non-Go consumers.

## Requirements

Go with cgo enabled and any C99 compiler. Linux, macOS and Windows (mingw);
the AVX2 multi-versioning engages automatically on x86-64 glibc systems and
falls back to portable code elsewhere.
