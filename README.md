# burn-probe

[![CI](https://github.com/bengtfrost/burn-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/bengtfrost/burn-probe/actions/workflows/ci.yml)

Burn 0.20 + CubeCL 0.9 on Intel Comet Lake GT2 (i5-10210U, UHD 620).
Benchmark probe and findings.

## Findings

- [f16 GEMV finding and no-go decision](docs/f16-gemv-finding.md) — medium GEMV 5.72 → 2.80 ms, LM head 124.1 → 47.3 ms; both beat AVX2.
- [vulkex bridge](docs/vulkex-writeup.md) — the mistral.rs sidecar that measured the same hardware from the hand-written-executor side.

## State

Tag: `v0.1-f16-gemv`

Status: educational; llama.cpp remains the production backend on this hardware. See the f16 finding for the reasoning.
