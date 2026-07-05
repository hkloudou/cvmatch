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
```

`Match` panics if an image is empty or `sub` is larger than `parent`, matching
the behaviour of the OpenCV-backed original.

## Results

Measured on a 4-core Intel Xeon @ 2.10 GHz (linux/amd64, Go 1.24) against
[`hkloudou/cv2`](https://github.com/hkloudou/cv2) v0.41200.0 (bundled static
OpenCV 4.12). Reproduce with `cd bench && go test -bench . -benchtime 5x`.

### Speed (`ns/op`, lower is better)

| scenario | cv2.Match | cvmatch.Match | speedup | cvmatch.MatchGray | speedup |
|---|---|---|---|---|---|
| 1280x720, 96x96 sub | 150.2 ms | **84.2 ms** | 1.8x | **33.1 ms** | 4.5x |
| 1920x1080, 128x128 sub | 379.9 ms | **152.1 ms** | 2.5x | **59.6 ms** | 6.4x |
| 1920x1080, 32x32 sub | 287.9 ms | **108.5 ms** | 2.7x | **46.0 ms** | 6.3x |
| 3840x2160, 256x256 sub | 1646.4 ms | **732.0 ms** | 2.2x | **302.0 ms** | 5.5x |

### Peak memory (VmHWM delta for one 1080p/128px match, fresh process)

| implementation | peak RSS above baseline |
|---|---|
| cv2.Match | ~145.5 MB |
| cvmatch.Match | **~14.8 MB** (9.8x less) |
| cvmatch.MatchGray | ~16.8 MB |

### Native code size (linux/amd64)

| artifact | size |
|---|---|
| cv2 bundled static libs (`libopencv_core.a` + `libopencv_imgproc.a` + zlib + wrapper) | 16.1 MB |
| `libcvmatch.a` | **30 KB** (~525x smaller) |
| minimal linked Go binary (`-ldflags "-s -w"`) | 5.82 MB (cv2) vs **1.57 MB** (cvmatch) |

### Accuracy

`Match` is numerically aligned with OpenCV: on the parity test in `bench/`,
`minVal` agrees to 6 decimal places, `maxVal`/locations are identical
(`cv2: min=-0.430550 max=1.000000 @(431,285)` vs
`cvmatch: min=-0.430550 max=1.000000 @(431,285)`), and the package's own tests
hold every response-map element within `1e-4` of a float64 brute-force
reference across single/multi-channel, strided, and degenerate inputs.

## How it works

The pipeline mirrors OpenCV's `matchTemplate(TM_CCOEFF_NORMED)`
(`modules/imgproc/src/templmatch.cpp` — the algorithm is identical in 4.8.1
and 5.x), then removes the parts that cost memory and time without changing
the output:

1. **Cross-correlation via block FFT** (`crossCorr` in OpenCV): the image is
   processed in tiles sized by OpenCV's own heuristic (`blockScale = 4.5`,
   `minBlockSize = 256`); the template spectrum is transformed once per
   channel; each tile is forward-transformed, multiplied by the conjugate
   spectrum and inverse-transformed. The FFT is a compact power-of-two
   radix-2 kernel: rows pack two real rows per complex transform (2-for-1
   real FFT), columns run batched across contiguous row elements so the
   butterflies auto-vectorize. Hot loops are compiled twice via
   `target_clones` (baseline x86-64 + AVX2/FMA) and selected at load time.
2. **Normalization** (`common_matchTemplate` in OpenCV): identical formulas,
   including the `min(0.5, 10*FLT_EPSILON*wndSum2)` rounding guard and the
   `1.125` saturation band — but instead of materializing two full
   double-precision integral images (`sum` + `sqsum`, ~132 MB for 1080p RGBA
   in OpenCV), window statistics come from O(width) sliding column sums.
3. **`minMaxLoc` is fused** into the normalization pass (same row-major
   first-occurrence tie semantics), so no extra scan and no extra buffer.
4. **Constant channels are skipped**: a channel that is constant within each
   image (the alpha plane of virtually every screenshot) contributes exactly
   zero to the CCOEFF numerator, denominator and template norm. `Match`
   detects this and processes 3 of 4 channels — the output is provably
   unchanged (covered by tests) and a quarter of the work disappears.
5. **Zero-copy inputs**: `*image.RGBA`, `*image.Gray` and the Y plane of
   `*image.YCbCr` (JPEG) are passed straight into C with their native
   strides — sub-images included, no redraw, no `Mat` copies.

Total transient native memory for a 1080p match is the `float32` response map
(~8 MB) plus a few hundred KB of FFT scratch, all freed before returning; the
Go side of `Match` allocates nothing beyond the return values.

## Layout

- `cvmatch.c` / `cvmatch.h` — the whole native implementation (C99, no deps).
- `cvmatch.go` — public API and zero-copy image conversion.
- `bench/` — separate module holding the `hkloudou/cv2` comparison
  (benchmarks, numeric parity test, `memprobe` peak-RSS tool, binary-size
  probes) so the root module stays dependency-free.
- `make lib` — builds the standalone `libcvmatch.a` for non-Go consumers.

## Requirements

Go with cgo enabled and any C99 compiler. Linux, macOS and Windows (mingw);
the AVX2 multi-versioning engages automatically on x86-64 glibc systems and
falls back to portable code elsewhere (`-DCVM_NO_TARGET_CLONES` to opt out).
