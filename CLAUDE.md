# CLAUDE.md — working rules for this repository

cvmatch implements OpenCV-compatible `TM_CCOEFF_NORMED` template matching
in pure Go with hand-written SIMD kernels. This file records the goals,
invariants and workflows that every change must follow.

## Goals, in priority order

1. **KISS** — as little code as possible; delete redundant code eagerly.
2. **Faster benchmarks** (the most important goal).
3. **Lower memory use** — but speed wins when the two conflict.
4. **OpenCV-consistent output** (owner directive, 2026-07-17: OpenCV is a
   black box f(x) — replicate its answers, not its code): extremum
   locations must match native `matchTemplate(TM_CCOEFF_NORMED)` +
   `minMaxLoc` on CV_8UC4/8UC1, and scores must stay within a **5%
   deviation budget** (the pipeline actually lands ~1e-6; every
   deliberate deviation from OpenCV's rounding order records its
   worst-case contribution so the sum stays ≪ budget). Bit-replication
   of OpenCV's internal rounding order is NOT required. Non-negotiable.
5. **Self-determinism.** Same input ⇒ bit-identical output across
   amd64/arm64, asm/purego builds and all thread counts. This is what the
   golden hashes, asm↔scalar parity and threads tests pin. Non-negotiable.
6. **No memory leaks or correctness regressions.** Non-negotiable.

## The determinism framework (never violate)

Until 2026-07-17 this framework additionally required bit-identity to
OpenCV itself; that requirement is retired (goal 4), the self-determinism
rules below remain. Any transformation must keep ONE fixed, exactly
specified op sequence shared by the scalar code and the asm kernels:

- **FMA is allowed** (it was banned only to mirror OpenCV's non-fused
  rounding), with one sharp caveat for float32 lanes:
  `float32(math.FMA(float64(a), float64(b), float64(c)))` **double-rounds**
  and does NOT always equal single-precision `VFMADDPS`/`fmla.4s`
  (counterexample: a=0x1.000006p0, b=1.5, c=2^-60). Scalar f32 fusion must
  go through a proven correctly-rounded `fma32` helper (round-to-odd
  repair of the f64 FMA), and the scalar code and the asm must fuse the
  **same** products (state the fusion points in the kernel contract; the
  asm↔scalar parity test then enforces the pairing bit-for-bit) — and
  the scalar sequence is what arm64 runs, so its cost there is the
  arm64 cost.
  f64 `math.FMA` needs no repair. Soft-float targets (386, riscv64, wasm)
  are slow but bit-identical — acceptable for fallback targets.
- **Forbidden instructions** (vendor-defined output bits break
  self-determinism even within one architecture): `VRSQRTPS`, `VRCPPS`,
  `frsqrte`/`frsqrts` chains, `frecpe`, x87 80-bit paths, and any
  FTZ/DAZ mode change.
- **Guard alignment**: the normalization guard/denominator chain
  (window sums, variance guards, tie decisions) stays fed by exact
  integers — only the FFT correlation numerator spends tolerance
  budget. This keeps near-degenerate windows (flat regions) stable no
  matter how the FFT is restructured.
- **Tolerance-budget ledger**: every merged deviation from OpenCV's op
  order records, in this file, what reorders, its worst-case bound, and
  the measured same-dump before/after delta; the running sum stays ≪ the
  5% budget. Ledger:
  - 7.1 argmin geometry (PR #18): tile DFT sizes differ from OpenCV's
    block heuristic → different FFT rounding paths; measured same-dump
    worst |Δ| ~2e-5.
  - 7.3 normalize-f32 (PR #19): float32 normalization tail (exact-integer
    cross/idiff unchanged); measured worst |Δ| 6.2e-4
    (window4k_button96x32) — the current suite-wide worst.
  - 7.4-lite last-band dftH shrink: only the FFT numerator reorders on
    short last bands (guard chain untouched); affected-scene |Δ| ~5e-6,
    suite worst unchanged.
  - 7.2a radix-4 columns: every column FFT reorders (radix-4 butterflies,
    odd stage moved to a twiddle-free bottom head); measured same-dump
    suite worst IMPROVED 6.2e-4 → 4.7e-4 (window scenes; the 7.3 f32
    tail still dominates the drift, the FFT contribution stays ~1e-5
    class).
- **Golden constants change only via the deliberate re-record flow**
  (`make regolden`, Phase 7.0): native tolerance parity must pass BEFORE
  recording, the commit log carries a `Goldens:` reason trailer, and the
  full matrix (both builds, race, the arm64 qemu leg) reproves
  self-identity on the new constants.
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
- **NTT stays dead on merit, under any contract**: exact integer
  convolution was built (`MatchExact`, v1.1.x tags) and **measured ~17x
  slower** than the float FFT path (64-bit modular butterflies have no
  SIMD; two-real-rows packing does not survive in Z_p — details in the
  README headroom section). Do not re-propose it absent genuinely new
  math; "integer arithmetic is exact and therefore desirable" was the
  intuition this measurement refuted.

## SIMD kernel workflow (`internal/simd`)

- The kernels are **default-on** (amd64 + gc only — the arm64 NEON
  twins were deleted by owner decision 2026-07-17; arm64 runs the scalar
  loops in every build mode); `-tags purego` (community-standard tag)
  opts out to 100% high-level Go with `simd.Enabled` a constant false so
  every kernel call site dead-code-eliminates. Both modes must stay
  bit-identical; the arm64 CI leg reproves the golden anchors on real
  hardware (that is the cross-arch self-determinism proof — scalar Go
  float code must keep its explicit float32()/float64() conversion
  barriers, or gc's arm64 contraction would fuse mul-adds and break it).
- Shared, arch-independent contracts live in `simd_kernels.go`; every
  kernel's doc comment states its exactness argument and bounds contract.
- The asm is split by domain: `simd_amd64.s` (CPU detection) +
  `simd_{fft,pack,norm}_amd64.s`. File-scoped `<>` symbols must live in
  the file that uses them.
- **Comment ratchet**: every kernel body annotates registers at first
  use; every NEW kernel additionally gets a register-map header block
  (see NormRow in simd_norm_amd64.s for the format), and whenever an
  existing kernel is touched, its map is added in the same change.
- Hand-written AVX2, Plan9 syntax, `NOSPLIT`, `VZEROUPPER` on every
  exit; runtime-detected via CPUID+XGETBV.
- Unimplemented shapes (e.g. packed-RGB step 3) must fall
  back to the scalar Go loops — gate call sites accordingly, and add a test
  for every fallback shape.
- Bounds discipline: wide loads may never read past a slice even when the
  buffer ends exactly at a page boundary — clamp vector trip counts by the
  slice length (see PackRows2) instead of assuming allocator slack.

## Validation gates (run all before pushing kernel/core changes)

```sh
go vet ./... && go vet -tags purego ./... && (cd bench && go vet ./...)
go test -count=1 .                    # default build (kernels + asm-vs-scalar parity)
go test -tags purego -count=1 .       # no-asm safe mode (golden anchors, scalar)
go test -race -count=1 . && go test -tags purego -race -count=1 .
GOOS=linux GOARCH=arm64 go test -c -o /tmp/t.arm64 . \
  && qemu-aarch64-static /tmp/t.arm64      # arm64 scalar suite under emulation
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
- **Fairness: every vs-native claim is 1T vs 1T.** OpenCV's
  matchTemplate is single-threaded, so charts and headline ratios never
  compare cvmatch multi-thread numbers against it. Internal threading
  stays in the product and is measured in the tables as its own fact.
- **Ratios only compare within one session on one machine.** Even the
  scalar/asm ratio shifts between runner microarchitectures (XEON vs
  EPYC measured 3.3-3.9x vs 4.1-4.2x for identical code), so judging an
  optimization requires an A/B on the same host, not two refreshes.
- **`bench-charts.yml` is the only benchmark pipeline** (weekly cron +
  `workflow_dispatch`, runnable on any ref; results publish to the
  `assets` branch, or stay a workflow artifact with `publish=false`).
  Regular CI runs correctness gates only; the ci.yml matrix steps
  execute only when ci.yml itself is dispatched (the quick
  perf-iteration loop, matrices tee'd into the job log for programmatic
  diffing).
- After merging a perf-relevant PR, dispatch `bench-charts.yml` to
  refresh the published numbers; routine merges do not re-measure.
- Profile before optimizing (`-cpuprofile` + `go tool pprof`); benchmark
  noise has repeatedly falsified plausible hypotheses in this repo.

## PR workflow

- **codex is observed, not waited for** (owner's standing instruction):
  Claude leads and merges on green CI when confident. Afterwards,
  periodically sweep recent merged PRs for codex comments and treat them
  as a self-check list — adjudicate late findings, fix forward the valid
  ones, politely rebut the rest.
- A codex **👍 reaction (on the PR body) means agreement** and fires no
  webhook; 👀 means still reviewing. Neither gates anything.
- Merge with a merge commit titled `Merge PR #N: <summary>`.
- Never write the literal skip-CI marker (bracketed `skip ci`) in commit
  messages — it suppresses all workflows on the PR. The bench-charts
  auto-commit uses it deliberately.
- Measured numbers in the PUBLISHED STORY live only in auto-generated
  artifacts (the README benchmatrix block and its summary paragraph) —
  never hand-write a live performance claim into README prose;
  reference the generated block instead. CLAUDE.md is the opposite
  case by design: its history and verdict ledgers MUST pin the measured
  basis of each decision (the determinism framework's ledger rule) —
  those are frozen audit records of past A/Bs, not live claims, and
  they do not go stale when the matrix refreshes.

## Measured-matrix pipeline

`bench-charts.yml` (weekly cron + dispatch): one amd64 job measures
native OpenCV (prebuilt static 4.12), both cvmatch builds, the parity
drift lines and peak RSS, renders via `docs/collect.py` (which also
writes `matrix.md` — markdown tables only, no charts by owner decision
2026-07-26), and **publishes `bench/{benchdata.json, matrix.md}` to the
`assets` branch** — main is never touched; the README links the stable
assets URLs. Dispatch with `publish=false` for a workflow artifact
instead (perf iteration). The comparison is the **amd64 three-way**:
native OpenCV + {asm, no-asm} × {Match, MatchGray}, 1T vs 1T, plus the
4T columns as the disclosed threading fact; arm64 runs the scalar path
and is correctness-tested in ci.yml, never benchmarked. No measured
number is ever hand-written — the generated summary paragraph carries
the ranges.

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
- Phase 4 made the assembly a global tag switch, and after the no-asm
  default measurably lost to native OpenCV on 8/14 scenes the owner
  flipped the polarity: kernels are default-on, `-tags purego` opts out
  (the kernels measure several-fold — exact ranges live in the generated
  summary), and the benchmark comparison became five-way (native + both
  builds on both architectures).
- Phase 5 cleaned the release flow: bench-charts (weekly + dispatch) is
  the only benchmark pipeline, CI is correctness-only (matrices run only
  when ci.yml is dispatched), the arm64 leg gained the same native
  OpenCV baseline as amd64 (identical comparison dimensions, one unified
  chart), and every measured number in the README is generated.
- Phase 6 closed the structural optimization space: the scalar fallback
  adopted the kernels' fused FFT shape; in-tile parallelism (row-pair
  distribution + width-chunked column-stage passes + chunked conjugate
  multiply) took single-tile scenes from ~1.0x to 1.2-1.5x at 4T on the
  reference machines. Remaining known levers and why they stayed
  unpicked under the old contract: SIMD prefix-sum spill lanes (low
  single-digit %), output-pruned inverse column FFT (<=15% niche, then
  needed a +-0 laundering proof), further NEON micro-tuning
  (diminishing). Two distinct kill categories, do not conflate them:
  split-radix / mixed-radix were blocked **by the old bit-identity
  contract only** (re-opened by the Phase 7 contract change below);
  **NTT was killed by measurement** — built as `MatchExact` (v1.1.x),
  ~17x slower than the float FFT path — and stays dead under any
  contract (see the determinism framework above).
- Phase 7 (in progress): the owner retired OpenCV bit-replication —
  "不复刻 OpenCV 的做法、只复刻它的答案": OpenCV is a black box f(x);
  matching locations + scores within a 5% budget is the contract,
  self-determinism stays non-negotiable (goals 4-5). A tolerance parity
  gate + deliberate golden re-record flow must land before the first
  deviating optimization ships. Ideas still answer to the same standard:
  A/B on the reference machines, above the noise floor. Shipped so far:
  7.0 verification framework + 7.1 tile-geometry argmin (PR #18 —
  reference A/B: amd64 native-normalized geomean +7%, arm64 +18%
  absolute on its homogeneous runners; Match VmHWM 48→35 MB) and 7.3
  normalize-f32 (PR #19 — exact-integer cross/idiff feeding a float32
  tail; one NormRow kernel for every cn plus the SpillStats1 cn=1 spill
  kernel; reference A/B: Match native-normalized geomean +6.7%, improved
  14/14 scenes, gray flat within noise, purego +7%; per-cn stats caps
  after a codex finding) and 7.4-lite last-band dftH shrink (the full
  per-edge-tile 7.4 was scoped down after an analytic scan showed the
  7.1 argmin already absorbs edge waste on 12/14 scenes; the shrunk
  template spectrum is a free stride-2^s row gather of tspec ×2^s via
  the exact decimation identity — the naive rebuild variant measured as
  a LOSS on asm and was discarded) and 7.2a radix-4 columns under the
  owner's full delegation (2026-07-26: "你可以都做对吧,你自己决定。我只
  验收结果"): the FMA leg died on day one (fma-in-pipeline in the
  verdict ledger), so the column passes run the no-FMA radix-4 engine
  (colsR4Go + FFTColsR4/FFTColsHead kernels; the retired radix-2 column
  passes and their FFTCols4/FFTColsBfly kernels were deleted). The odd
  stage moved to a twiddle-free bottom head. Local interleaved 20x e2e:
  asm geomean +2.2% (no scene regressed), purego +5.2% (uniform) — the
  purego win is also arm64's. Rows stay on the radix-2 fftGo/FFTStages
  pair: 7.2b (row flip to fftR4, which already exists in-tree) is the
  remaining optional leg, to be judged the same way after 7.2a's
  reference verdict; re-judge sweep-fusion (PR #21 PARKED) only after
  that.

- Phase 8: the owner deleted the arm64 NEON backend outright ("全面删除
  arm64 的汇编") — ~4.8k lines (generated stream + `_gen` clang/objdump
  generator + plumbing) removed; arm64 degrades to the scalar loops in
  every build mode and keeps only the correctness CI leg (goldens on
  real arm64 hardware = the cross-arch self-determinism proof). The
  benchmark story is the amd64 three-way: native OpenCV vs asm vs
  no-asm. Consequence for 7.2: a radix-4+FMA rewrite now needs only ONE
  asm backend, but the scalar fma32 cost lands directly on arm64.

## Phase 7 design-study verdict ledger (adjudicated 2026-07-17)

- **fma-everywhere / fma-in-pipeline**: DEAD BY SCALAR COST (measured
  2026-07-26) — hardware f32 FMA in asm forces the correctly-rounded
  fma32 scalar twin into the purego/arm64 leg (self-determinism), and
  the radix-4+fma32 scalar measured **3.8x slower** than the plain
  radix-4 scalar (27.4µs vs 5.7µs at n=512 1-D; fma32 is ~10 sequenced
  f64 ops per fusion). No asm gain can buy back a 3-4x purego FFT
  regression under the owner's two-build positioning. The fma32 helper
  + oracle tests stay in-tree (commit be911ae) as the framework's
  reference implementation should the contract ever change. Same kill
  class as NTT: do not re-propose absent new math.
- **direct-int-corr**: PARKED (no demonstrated workload) — correctness
  clean, but 0% on the published suite (the smallest bench template,
  24x24, already loses to the AVX2 FFT baseline by its own arithmetic);
  revisit only if the owner confirms sub-16x16 templates matter.
- **pruned-inverse**: REJECTED (do not build) — 0.1-2.1% e2e is under
  the noise floor, and per-edge-tile DFT sizing (7.4) strictly dominates
  it. Retained knowledge: inverseRowPair unconditionally reads spec row
  r+1 — a trap for any future row-band pruning.
- **rsqrt-approx**: REJECTED (breaks determinism) — estimate-instruction
  bits are vendor-defined even within amd64; sub-noise gain; the
  instructions are on the forbidden list above.
- **column-FFT octet fusion (3 stages/sweep)**: PARKED PENDING 7.2
  (PR #21, closed unmerged) — local 20x interleave said x1.030 geomean,
  but the reference machines arbitrated it away: suite geomean
  x0.995/x0.983 across two dispatch pairs, and the per-scene signal
  split into a consistent 1080p-slab win (+3~9%) vs a consistent
  4k-slab LOSS (−2~−5%) — the L3-vs-DRAM regime boundary that a fixed
  `n·width·8 > 1MB` gate cannot encode portably (the same scenes
  measured +7% on the local EPYC container: gate constants tuned
  off-reference do not transfer). Retained: judge sweep-fusion again
  only as a radix-4 epilogue after 7.2 lands, with a two-sided gate if
  the reference machines still show column-pass dominance. Use the
  purego columns of any dispatch-pair A/B as a null control — pair 2 of
  this study showed +9~34% on byte-identical purego code and was
  discarded.
- **split-radix**: DEAD ON MERIT — ~6% FFT-op delta vs radix-4 is
  sub-noise end-to-end, for 2.5-3x kernel surface and a broken
  pass/barrier structure. Sits next to NTT: do not re-propose without
  new math.
- **one-big-FFT**: DEAD ON MEASUREMENT — 1.13-1.53x slower with 3-6x
  memory; the 7.1 argmin converges to one tile exactly where that is
  optimal anyway.
- **mixed-radix 2^a·3^b·5^c**: DEAD ON MERIT — padding waste only
  exists in the one-big regime; +60-100% kernel surface for single-digit
  average gain.
- **NTT**: measured 17x slower (see the determinism framework) — dead
  under any contract.
