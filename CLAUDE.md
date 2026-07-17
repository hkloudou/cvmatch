# CLAUDE.md — working rules for this repository

cvmatch implements OpenCV-compatible `TM_CCOEFF_NORMED` template matching
in pure Go with hand-written SIMD kernels. This file records the goals,
invariants and workflows that every change must follow.

## Goals, in priority order

1. **KISS** — as little code as possible; delete redundant code eagerly.
2. **Faster benchmarks** (the most important goal).
3. **Lower memory use** — but speed wins when the two conflict.
4. **Bit-identical output to native OpenCV** `matchTemplate(TM_CCOEFF_NORMED)`
   + `minMaxLoc` on CV_8UC4/8UC1. Non-negotiable.
5. **No memory leaks or correctness regressions.** Non-negotiable.

## The bit-exactness framework (never violate)

Any transformation is legal only if it provably preserves every output bit:

- Float ops must keep the scalar code's **single-rounding order**. No FMA,
  ever (Go asm: never `VFMADD*`/`fmla`; the gc compiler does not contract
  on amd64/arm64 for the explicitly float32-barriered expressions used here).
- Exact IEEE identities may be used: addition commutes bitwise;
  `x-y ≡ x+(-y)` with negation by sign-bit xor; multiplication by exact
  powers of two; `sqrt`/`div` are correctly rounded on both ISAs; ±0 sign
  differences are acceptable only where a later guard provably launders them
  (document the laundering argument in a comment).
- **Integer arithmetic reassociates freely** — the sliding column sums and
  window statistics are exact integers below 2^53, so any vector regrouping
  or factorization (e.g. `(a-b)(a+b)` for `a²-b²`) is bit-safe.
- int64→float64 conversions are exact below 2^52 (amd64 uses the
  2^52 or/sub identity, arm64 `scvtf`).
- Extrema scans must keep OpenCV's **first-occurrence tie rule**; vector
  scans achieve it via the lexicographic (value, index) tournament.
- The deterministic `sincospi` twiddle generator is shared by all paths —
  never call libm/math trig in the hot pipeline.

## SIMD kernel workflow (`internal/simd`)

- The kernels are **opt-in**: they compile only under `-tags cvmatch_asm`
  (on amd64/arm64 + gc). The default build is 100% high-level Go — the
  safe mode — with `simd.Enabled` a constant false so every kernel call
  site dead-code-eliminates. The SIMD build is worth ~3-4x end to end;
  both modes must stay bit-identical and both run in CI on both arches.
- Shared, arch-independent contracts live in `simd_kernels.go`; every
  kernel's doc comment states its exactness argument and bounds contract.
- **amd64**: hand-written AVX2 in `simd_amd64.s` (runtime-detected;
  Plan9 syntax, `NOSPLIT`, `VZEROUPPER` on every exit).
- **arm64**: Go's assembler has no un-fused vector FP ops, so bodies are
  written as annotated ARM64 assembly in `_gen/kernels.S` and spliced into
  `simd_arm64.s` as WORD streams by `_gen/gen.py` (clang cross-assembles;
  relocation-free constants only; registers x0-x15/v0-v31, no stack; each
  body ends in a single `ret`). Regenerate with `go generate ./internal/simd`;
  CI diffs the regenerated stream, so never edit `simd_arm64.s` by hand.
- Unimplemented shapes (e.g. packed-RGB step 3, cn=2 normalize) must fall
  back to the scalar Go loops — gate call sites accordingly, and add a test
  for every fallback shape.
- Bounds discipline: wide loads may never read past a slice even when the
  buffer ends exactly at a page boundary — clamp vector trip counts by the
  slice length (see PackRows2) instead of assuming allocator slack.

## Validation gates (run all before pushing kernel/core changes)

```sh
go vet ./... && go vet -tags cvmatch_asm ./... && (cd bench && go vet ./...)
go test -count=1 .                    # default build (golden anchors, scalar)
go test -tags cvmatch_asm -count=1 .  # SIMD build (kernels + asm-vs-scalar parity)
go test -race -count=1 . && go test -tags cvmatch_asm -race -count=1 .
GOOS=linux GOARCH=arm64 go test -tags cvmatch_asm -c -o /tmp/t.arm64 . \
  && qemu-aarch64-static /tmp/t.arm64      # NEON suite under emulation
GOOS=linux GOARCH=arm64 go test -c -o /tmp/tg.arm64 . \
  && qemu-aarch64-static /tmp/tg.arm64     # arm64 default build too
# TestGoldenOutputs passing on both architectures in both modes IS the
# bit-identity proof (the constants pin every output bit)
for t in linux/386 linux/riscv64 windows/amd64 wasip1/wasm darwin/arm64; do
  GOOS=${t%/*} GOARCH=${t#*/} go build ./...; done
```

`TestSIMDMatchesScalar` (bit-for-bit asm↔scalar), the kernel unit tests
(tie-breaking, saturation), and the golden-output tests are the correctness
anchors; the bench module's CI jobs additionally pin the public API
element-wise against a native OpenCV C++ binary.

## Benchmarking rules

- The **CI runners are the reference machines** (ubuntu-latest amd64,
  ubuntu-24.04-arm). Local/container numbers drift between sessions — never
  compare absolute numbers across hosts; compare only within one run.
- **1T numbers are the primary signal**; 4T on shared runners carries
  ±2-5% noise and shows outliers in both directions.
- Iterate on asm perf by pushing to the branch and dispatching
  `ci.yml` (`workflow_dispatch`); the arm64 matrices are tee'd into the
  job log so they can be diffed programmatically.
- Profile before optimizing (`-cpuprofile` + `go tool pprof`); benchmark
  noise has repeatedly falsified plausible hypotheses in this repo.

## PR workflow

- Every PR gets a dual review with **codex** (auto-reviews new PRs). Claude
  leads the review and does the merging; codex findings are suggestions to
  adjudicate, not directives — fix the valid ones, politely rebut the rest,
  resolve threads.
- A codex **👍 reaction (on the PR body) means approval** and fires no
  webhook — poll reactions/reviews when waiting. 👀 means still reviewing.
- Merge with a merge commit titled `Merge PR #N: <summary>`.
- Never write the literal skip-CI marker (bracketed `skip ci`) in commit
  messages — it suppresses all workflows on the PR. The bench-charts
  auto-commit uses it deliberately.
- After merging anything to `main`, the `bench-charts` workflow re-measures
  both architectures and commits refreshed charts + README matrix; verify
  it ran and that README prose still agrees with the new numbers.

## Charts pipeline

`bench-charts.yml` (push to main): a `bench-arm64` job uploads raw
benchmark output as an artifact; the `charts` job measures amd64 + native
OpenCV, then `docs/collect.py` merges everything into `docs/benchdata.json`
and `docs/genchart.py` renders the SVGs and rewrites the README between
the `benchmatrix` markers. The comparison is five-way: native OpenCV,
then {amd64, arm64} × {SIMD build, default build}, each with Match and
MatchGray. Keys: `asm*`/`agray*` = `-tags cvmatch_asm`, `go*`/`gray*` =
default build, `A` suffix = arm64. Keep chart series colors
meaning-stable across all charts (green = native, blue family = Match,
orange family = MatchGray; solid = SIMD build, light = default).

## History (context for future work)

- Phase 1 gave the pure-Go core full amd64 AVX2 kernel coverage; it now
  outperforms the former cgo/C core on every amd64 scene.
- Phase 2 added the arm64 NEON twins (bit-identical across architectures,
  proven by whole-map hashes) and the arm64 CI leg.
- Phase 2.5 tuned NEON to parity-or-ahead of the C core on arm64.
- Phase 3 removed the cgo C core entirely (it had become the slower path
  on both architectures); the golden-output tests recorded while both
  cores existed remain the bit-identity anchor, alongside the native
  OpenCV parity jobs in `bench/`.
- Phase 4 made the assembly a global opt-in switch (`-tags cvmatch_asm`):
  the default build is pure high-level Go (the safe mode, ~3-4x slower),
  and the benchmark comparison became five-way (native + both builds on
  both architectures).
