//go:build gc

package simd

// NEON backing for the kernels declared in simd_kernels.go. Advanced SIMD
// is architecturally guaranteed on AArch64 Linux/macOS/Windows targets,
// and the kernels avoid fused multiply-adds by construction (Go's
// assembler lacks un-fused vector FP ops, so the bodies are generated
// WORD streams — see _gen/kernels.S). AArch64 keeps IEEE semantics by
// default (FPCR.FZ=0, round-to-nearest), so the kernels are bit-identical
// to the scalar loops, as on amd64.

//go:generate python3 _gen/gen.py

// Enabled reports whether the NEON kernels can run (a var so tests can
// toggle the SIMD/scalar comparison; always true on arm64).
var Enabled = true
