//go:build !cgo

package cvmatch

// Impl reports which core is compiled in: "purego" — the dependency-free
// Go port used automatically when cgo is unavailable (CGO_ENABLED=0,
// cross-compilation without a C toolchain, etc.).
const Impl = "purego"

func matchU8(img []uint8, istride, iw, ih int, tpl []uint8, tstride, tw, th, cn, step int, result []float32) (float32, int, int, float32, int, int) {
	return matchU8Go(img, istride, iw, ih, tpl, tstride, tw, th, cn, step, result)
}
