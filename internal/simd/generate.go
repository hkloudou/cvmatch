package simd

// The arm64 NEON kernel bodies in simd_arm64.s are generated from
// _gen/kernels.S by _gen/gen.py (clang cross-assembles, so any host
// GOARCH can regenerate). The directive lives in this unconstrained file
// so plain `go generate ./internal/simd` finds it from any platform —
// build-constrained files are skipped by go generate's file scan.

//go:generate python3 _gen/gen.py
