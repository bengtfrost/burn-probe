# vulkex — bridge from the f16 GEMV finding

**Canonical source:** [bengtfrost/mistral.rs @ `vulkex-v1`](https://github.com/bengtfrost/mistral.rs/tree/vulkex-v1/crates/vulkex)
**Authoritative docs (in that repo):** [`vulkan-design-decisions.md`](https://github.com/bengtfrost/mistral.rs/blob/vulkex-v1/docs/vulkan-design-decisions.md), [`vulkan-device-report.md`](https://github.com/bengtfrost/mistral.rs/blob/vulkex-v1/docs/vulkan-device-report.md), [`vulkan-upstream-plan.md`](https://github.com/bengtfrost/mistral.rs/blob/vulkex-v1/docs/vulkan-upstream-plan.md)

This document does **not** restate vulkex's design analysis. It exists
to explain what vulkex measured, and why its result and the burn-probe
f16 finding are the same conclusion reached from opposite directions.

## What vulkex is

A standalone Vulkan compute executor sidecar for mistral.rs. Not a
Candle `BackendStorage`, not wired into `mistralrs-core`, not reachable
from the CLI. It exposes `Session::gemv` and `Session::gemm`, validated
against f64 CPU references, and benchmarks them on the same hardware
as this repo: Intel UHD 620 (Comet Lake GT2), i5-10210U, 16 GB shared
LPDDR4, Mesa / ANV.

The point was to answer the "should we build a full Candle Vulkan
backend?" question for one-twentieth of the cost. It did.

## Measured numbers

Prefill GEMM, M = 512. CPU reference is `matrixmultiply::sgemm`:

| Shape | CPU | Vulkan | Ratio |
|---|---:|---:|---:|
| 512×6144×2048 | 63.7 GFLOPS | 33.1 GFLOPS | 0.52× |
| 512×2048×6144 | 65.9 | 33.0 | 0.50× |
| 512×2048×2048 | 65.6 | 32.8 | 0.50× |
| 512×151936×2048 | 68.9 | 33.0 | 0.48× |

Decode GEMV, M = 1. Vulkan effective bandwidth:

| Shape | Vulkan GB/s |
|---|---:|
| 2048×2048 | 3.8 |
| 2048×6144 | 11.0 |
| 6144×2048 | 8.2 |
| 151936×2048 | 13.1 |

CPU backend on the same shapes streams weights at **19.3 GB/s**
(Qwen3-1.7B F32, `mistralrs bench --cpu`). Weighted model-level
prediction: **~1.1 T/s decode vs 2.4 T/s CPU**.

## Three structural gaps

From [`vulkan-design-decisions.md`](https://github.com/bengtfrost/mistral.rs/blob/vulkex-v1/docs/vulkan-design-decisions.md):

1. **Per-op dispatch overhead.** ~3 ms fixed cost per 2048×2048 GEMV
   × 112 dispatches/token ≈ 336 ms/token before any weight is read.
   38% of the decode total.
2. **No Q4_K dequant-fused kernel.** The FP32 path cannot load
   quantized GGUFs without inflating a 5 GB Q4_K_M file to 36 GB FP32.
3. **No subgroup reductions, no cooperative matrices.** Gen9.5 does not
   expose `VK_KHR_cooperative_matrix`; the shared-memory + `barrier`
   path is used unconditionally.

None of these are shader bugs. They are architectural choices that
llama.cpp's `ggml-vulkan` backend makes differently.

## How this intersects the f16 GEMV finding

Both projects measured Vulkan compute on the same hardware, from
opposite starting points:

| Question | vulkex | burn-probe |
|---|---|---|
| Numerically correct on Gen9? | Yes | Yes (f32 + f16) |
| Beats CPU on prefill? | No (0.48–0.52×) | No (0.23–0.52×) |
| Beats CPU on decode? | No in FP32 (~0.46×) | **Yes in f16** on two shapes (0.85× and 0.56× of AVX2 time) |
| Dominant gap? | Dispatch + no Q4_K | Kernel quality + no Q4_K |
| Production backend? | llama.cpp | llama.cpp |

The f16 result does not contradict vulkex. It shows that **the decode
axis is partially winnable** — halving bytes overcomes the CPU's
bandwidth advantage on the two largest shapes — while prefill is not,
because prefill is Q4_K- and MMA-bound, and neither burn-probe nor
vulkex closed those gaps.

## Why this matters

Without vulkex, "Burn is 2× slower than llama.cpp on prefill" reads as
"Burn is bad." With vulkex, the correct reading is: **even a hand-
written FP32 executor using the full Vulkan API and no framework
overhead is 2× slower than CPU on this hardware.** The gap is dtype
(Q4_K) and dispatch architecture, not framework quality.

That is the fact that makes the burn-probe no-go recommendation
defensible on the merits rather than as a sigh.

## Reopen conditions

Unchanged from vulkex's own analysis, see
[`vulkan-design-decisions.md` §1](https://github.com/bengtfrost/mistral.rs/blob/vulkex-v1/docs/vulkan-design-decisions.md):

- A measured ≥1.5× CPU prefill win with a graph-batched, Q4_K-native
  executor on this or comparable hardware.
- Candle upstreams a Vulkan or ROCm backend.
- **A different device** with real fp16 throughput and 200+ GB/s
  memory bandwidth. This is the only condition that changes the
  hardware conclusion, not just the implementation conclusion.
