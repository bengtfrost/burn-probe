# f16 GEMV on Intel Comet Lake GT2 — finding and no-go decision

**Hardware:** i5-10210U (4C/8T, AVX2+FMA, no AVX-512), Intel UHD 620
(Comet Lake GT2, Gen9, 184 SPs, no Tensor Cores, no cooperative
matrix), 16 GB shared LPDDR4 @ ~25.6 GB/s, Mesa 26.2.5 / ANV,
Vulkan 1.4.354. Void Linux 6.18.58, Rust 1.99.0, Burn 0.20.1,
CubeCL 0.9.0.

**Date:** 2026-10-26. Tags: `v0.1-f16-gemv`.

**Predecessor finding:** [`docs/vulkex-writeup.md`](vulkex-writeup.md) — a hand-written FP32 Vulkan executor on the same hardware reached the same production conclusion from the opposite direction.

## Summary

Burn-Vulkan f16 GEMV beats both f32-Vulkan and a hand-written AVX2
GEMV on the two shapes that dominate single-token decode. The win
is entirely from halved bytes against a driver-imposed bandwidth
ceiling — not from better compute throughput on fp16, which Gen9
exposes at fp32 rate. Prefill GEMMs remain structurally behind
llama.cpp and are not addressable at the Burn app layer.

**Production decision: llama.cpp remains first choice.**
Burn/Vulkan is documented here as an educational reference state,
not a migration path.

## The f16 GEMV result

Measured cold, one shape per run, `--sample-count 30 --sample-size 1`
(3/1 for the LM head), after `sleep 300` between runs. `intel_gpu_top`
in a second terminal during each run. Median of 30 samples, or of 3
for the LM head.

| shape (m, k) | f32 | f16 | AVX2 | f16 / AVX2 |
|---|---:|---:|---:|---:|
| (2048, 2048) | 4.20 ms | 2.69 ms | 1.04 ms | 2.59× |
| (2048, 6144) | 6.28 ms | 3.81 ms | 3.35 ms | 1.14× |
| **(6144, 2048)** | **5.72 ms** | **2.80 ms** | **3.31 ms** | **0.85×** |
| **(151936, 2048)** LM head | **124.1 ms** | **47.3 ms** | **84.9 ms** | **0.56×** |

The two largest shapes flip from **1.73× and 1.46× slower than CPU**
(f32) to **1.18× and 1.79× faster than CPU** (f16). This is the
first workload in the project where Burn-Vulkan beats a proper AVX2
reference.

### Why f16 wins

`intel_gpu_top` Render/3D during the f32 medium GEMV reads ~44%,
with brief spikes to 90%. That is *not* compute saturation. It is
the GPU waiting on the memory subsystem. Compute is idle; bandwidth
is the bottleneck.

Effective bandwidth, computed as (bytes touched) / (wall time):

| shape | f32 | f16 | AVX2 |
|---|---:|---:|---:|
| (6144, 2048) | 8.8 GB/s | 9.0 GB/s | 15.2 GB/s |
| LM head | 10.0 GB/s | 13.2 GB/s | 14.7 GB/s |

Two observations:

1. **f32 and f16 hit the same ~9 GB/s on the medium GEMV.** Halving
   bytes therefore halves wall time. This is pure bandwidth
   arithmetic — nothing about f16 "helps compute" on Gen9, which
   exposes fp16 at fp32 rate per ANV.

2. **The f16 LM-head kernel actually reaches 13.2 GB/s** against
   the f32 kernel's 10.0 GB/s. That is *not* just byte-halving; the
   f16 kernel path in CubeCL uses better coalescing or tiling than
   the f32 path on this shape. This is why f16 improves the LM head
   by 2.62× rather than the 2.0× that pure byte-halving would
   predict.

### What it does not reach

AVX2 sits at ~15 GB/s on both shapes. The GPU's f16 kernel reaches
13.2 GB/s on the LM head, so it is still ~13% short of CPU on raw
memory efficiency. The f16 GPU wins only because it moves half the
bytes at that slightly lower efficiency. On a device where AVX2 is
not already present, the GPU would win by more.

Against llama.cpp: on the LM head alone, llama.cpp's Q4_K_M weights
move ~4× fewer bytes than f16, at comparable effective bandwidth.
llama.cpp remains ~2× faster on that operation even after this
result. The dtype ladder is f32 → f16 → Q4; this session moved up
one rung, not two.

## The large-GEMM diagnostic

`burn_gemm_lm` for shape `(512, 151936, 2048)` runs at 16.7–18.6 s
on the GPU (18.55 s at the cold 2026-10-24 rebaseline, 16.7 s at
reduced sample count). CPU `mm_gemm` on the same shape: ~4.2 s.
Ratio: **4.4× slower than CPU** — the worst regression in the
project.

With `env_logger` installed and `RUST_LOG=cubecl=debug,info`, the
autotuner's selection is now visible:

    MatmulEntry<
      TensorArgs,
      SimpleMatmulFamily<
        UnitMatmulFamily<RegisterMatmul, StridedStageFamily, FilledStageFamily>,
        SyncFullCyclicLoading<ColMajorTilingOrder>,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        UnitWriterFamily
      >,
      RowMajorGlobalPartitionMatmul,
      WgpuRuntime
    >

Decompose against CubeCL's family hierarchy:

| Component | Selected | Meaning |
|---|---|---|
| Family | `SimpleMatmulFamily` | Single-stage, no pipelining |
| Stage tile | `UnitMatmulFamily` | 1×1 per thread, no register blocking |
| Load | `SyncFullCyclicLoading` | Barrier per K-stage boundary |
| Writer | `UnitWriterFamily` | One thread per output element |

This is the simplest available GEMM strategy. Retested with
`--sample-count 10` — same line, bit-for-bit. **This is autotuner
preference for the shape, not a sample-count artefact.**

### The kernel is not memory-bound; it is barrier-bound

Global B traffic is ~20 GB (B re-read ~16× if the M-tile height is
32 rows). At 25.6 GB/s that is 0.8 s. Observed is 16.7 s —
**~20× over the memory bound.**

Effective throughput: 318.6 GFLOP / 16.7 s = 19.1 GFLOP/s, or
about 4.2% of Gen9's f32 FMA peak (~460 GFLOP/s, matching
llama.cpp's prefill number on the same hardware).

`intel_gpu_top` shows **Render/3D at 100%** during the run. The
shaders are *issuing* — but each useful FMA is amortized over
20–40 cycles of shared-memory load latency and `barrier`
synchronization. High occupancy, low IPC. That is what 4.2% of peak
looks like.

### Fixable at the app layer?

No. CubeCL selects the family; Burn does not expose a strategy
override. A better kernel (`DoubleBuffered`, register-blocked tiles
with larger `tile_m`) exists in CubeCL's abstract family set, but
the autotuner does not choose it for this shape, and forcing it
requires patching CubeCL 0.9 — out of scope by the project's
working method.

### Worth fixing?

No. Even landing at CPU parity (4.2 s) would match `mm_gemm`, but
still lose to llama.cpp's Q4_K_M Vulkan prefill because the dtype
gap dominates. The regression is documented, not actionable.

## Measurement methodology

The session's first attempt on 2026-10-24 produced numbers that were
later found to be thermally contaminated. The correction is now the
mandatory protocol for any future measurement in this repo:

1. `pgrep -af llama-server` must be empty.
2. `free -h` must show ≥4 GB available.
3. `sleep 300` after any prior GPU work.
4. **One shape per benchmark invocation.** Mixing f32 and f16 runs in
   a single `cargo bench` call allows thermal drift to inflate later
   runs by 1.5–3.8× — invisible in the output, only detectable by
   comparing shapes that ran earlier in the same invocation to their
   cold baseline.
5. `intel_gpu_top` in a second terminal. Watch `GT freq` and
   `Render/3D`. If `GT freq` drops from ~1100 to ~300–400 MHz, the
   run is throttled and unusable.
6. `--sample-count 30 --sample-size 1` for medium GEMV,
   `--sample-count 3 --sample-size 1` for LM head. The split keeps
   total GPU time per run under ~10 s, below the throttle knee for
   this 15 W package.

The 2026-10-24 f16 log is retained as a **negative reference**: it
shows f16 winning directionally even under thermal handicap, but
every GPU number in it is inflated by a non-uniform factor and must
never be cited as a measurement.

## Files and commits

Committed on `master`, tag `v0.1-f16-gemv`:

    baseline: f32 Vulkan GEMV/GEMM + AVX2 + mm references
    perf(gemv): add f16 Vulkan GEMV on Intel Gen9
    chore(gitignore): ignore bench-*.log and test-*.log artifacts
    diag(bench): install env_logger so RUST_LOG reaches cubecl; capture selection

Commit messages carry the measured numbers so they survive the
gitignored logs.

## Conclusion

Burn-Vulkan f16 is a genuine, measurable win on the two largest
GEMV shapes on Gen9. It closes the decode gap to CPU and, on the
LM-head shape, exceeds it. It does not touch prefill — prefill GEMMs
on this hardware are 0.23–0.52× CPU for Burn and 4.11× for
llama.cpp, a gap of dtype (Q4 vs f32/f16) and of MMA availability
that no Burn-layer change can close.

**Production: llama.cpp.** This work is a valid educational
reference for the limits of a f32/f16 CubeCL kernel stack on a
15 W Gen9 iGPU, not a migration path.

Future work on this repo should stop here on this machine.
The interesting next question is a different device — how much does
the f16-vs-f32 GEMV result hold on a GPU with real fp16 throughput
and 200+ GB/s memory bandwidth. That is a hardware swap, not
another tuning round on Comet Lake.
