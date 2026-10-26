// burn-probe: compare Burn's Vulkan backend against matrixmultiply (CPU) and
// llama.cpp's ggml-vulkan on the same machine (Intel CML GT2 / i5-10210U).
//
// Shape provenance: crates/vulkex/tests/bench_shapes.rs (Qwen3-1.7B).
// dtype: f32 on both sides. See vulkex/README for the vulkex × CPU numbers
// quoted in the project writeup.

use burn::tensor::{Distribution, Tensor, f16};
use burn_wgpu::Vulkan;
use divan::{Bencher, Divan};

type B = Vulkan<f32, i32>;
type Bf16 = Vulkan<f16, i32>;

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!(
            "warning: AVX2/FMA not detected; gemv_avx2_ref will fail if run. \
             Re-run with `-- gemm|mm_` to skip it."
        );
    }
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    Divan::from_args().run_benches();
}

// ---- GEMV shapes: w[m,k] @ x[k,1] -> y[m,1] ----
const GEMV_SHAPES: &[(usize, usize)] = &[
    (2048, 2048), // attention projections
    (2048, 6144), // MLP up
    (6144, 2048), // MLP down
];
const GEMV_LM_HEAD: &[(usize, usize)] = &[
    (151936, 2048), // LM head
];

// ---- GEMM shapes: a[m,k] @ b[k,n] -> c[m,n] ----
const GEMM_SHAPES: &[(usize, usize, usize)] =
    &[(512, 6144, 2048), (512, 2048, 6144), (512, 2048, 2048)];
const GEMM_LM_HEAD: &[(usize, usize, usize)] = &[
    (512, 151936, 2048), // LM head
];

// ============================================================
// Burn-Vulkan
// ============================================================
//
// Burn's matmul returns a lazy handle. Without forcing a device sync inside
// the timed closure, divan measures kernel submission (~µs) rather than
// execution (~ms). `sync()` reduces to a scalar and reads 4 bytes back to the
// host — the smallest fence wgpu exposes for "wait for all preceding kernels
// to finish".

fn sync(t: Tensor<B, 2>) -> f32 {
    t.sum().into_scalar()
}

fn sync_f16(t: Tensor<Bf16, 2>) -> f16 {
    t.sum().into_scalar()
}

#[divan::bench(args = GEMV_SHAPES)]
fn burn_gemv(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<B, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync(w.clone().matmul(x.clone()));
    bencher.bench_local(|| sync(w.clone().matmul(x.clone())));
}

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 3, sample_size = 1)]
fn burn_gemv_lm(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<B, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync(w.clone().matmul(x.clone()));
    bencher.bench_local(|| sync(w.clone().matmul(x.clone())));
}

// f16 GEMV probes: same shapes as burn_gemv / burn_gemv_lm, halved bytes.
// The open question (handoff Q3) is whether these are bandwidth-bound enough
// that halving traffic beats the f32 baseline. If yes, Gen9 fp16 can pay off
// even though ANV exposes fp16 at the same compute rate as fp32.

#[divan::bench(args = GEMV_SHAPES)]
fn burn_gemv_f16(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<Bf16, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<Bf16, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync_f16(w.clone().matmul(x.clone()));
    bencher.bench_local(|| sync_f16(w.clone().matmul(x.clone())));
}

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 3, sample_size = 1)]
fn burn_gemv_lm_f16(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<Bf16, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<Bf16, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync_f16(w.clone().matmul(x.clone()));
    bencher.bench_local(|| sync_f16(w.clone().matmul(x.clone())));
}

#[divan::bench(args = GEMM_SHAPES)]
fn burn_gemm(bencher: Bencher, &(m, n, k): &(usize, usize, usize)) {
    let device = Default::default();
    let a: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random([k, n], Distribution::Default, &device);
    let _ = sync(a.clone().matmul(b.clone()));
    bencher.bench_local(|| sync(a.clone().matmul(b.clone())));
}

#[divan::bench(args = GEMM_LM_HEAD, sample_count = 3, sample_size = 1)]
fn burn_gemm_lm(bencher: Bencher, &(m, n, k): &(usize, usize, usize)) {
    let device = Default::default();
    let a: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random([k, n], Distribution::Default, &device);
    let _ = sync(a.clone().matmul(b.clone()));
    bencher.bench_local(|| sync(a.clone().matmul(b.clone())));
}

// ============================================================
// Burn-Vulkan LM head diagnostics
// ============================================================

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 10, sample_size = 1)]
fn burn_gemv_lm_probe(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<B, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync(w.clone().matmul(x.clone()));
    bencher.bench_local(|| sync(w.clone().matmul(x.clone())));
}

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 3, sample_size = 1)]
fn burn_gemv_lm_batched(bencher: Bencher, &(m, k): &(usize, usize)) {
    let device = Default::default();
    let w: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let x: Tensor<B, 2> = Tensor::random([k, 1], Distribution::Default, &device);
    let _ = sync(w.clone().matmul(x.clone()));

    const N: usize = 5;
    bencher.bench_local(|| {
        // Element-wise add forces all N matmuls to materialize (Burn DCEs
        // unused intermediates). The N-1 additions cost ~3 MB of traffic
        // against 1.24 GB per matmul — under 1% pollution.
        let mut total = w.clone().matmul(x.clone());
        for _ in 1..N {
            total = total + w.clone().matmul(x.clone());
        }
        sync(total)
    });
}

// ============================================================
// matrixmultiply reference (CPU baseline, as used by vulkex)
// ============================================================

fn mm_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn setup_gemv(m: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = (0..m * k)
        .map(|i| {
            ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) as f32) / (u32::MAX as f32)
                - 0.5
        })
        .collect();
    let x: Vec<f32> = (0..k)
        .map(|i| {
            ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) as f32) / (u32::MAX as f32)
                - 0.5
        })
        .collect();
    let y = vec![0.0f32; m];
    (w, x, y)
}

fn setup_gemm(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 13) as f32 * 0.01).collect();
    let c = vec![0.0f32; m * n];
    (a, b, c)
}

#[divan::bench(args = GEMV_SHAPES)]
fn mm_gemv(bencher: Bencher, &(m, k): &(usize, usize)) {
    let (w, x, mut y) = setup_gemv(m, k);
    bencher.bench_local(|| {
        mm_ref(&w, &x, &mut y, m, 1, k);
        std::hint::black_box(&y);
    });
}

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 3, sample_size = 1)]
fn mm_gemv_lm(bencher: Bencher, &(m, k): &(usize, usize)) {
    let (w, x, mut y) = setup_gemv(m, k);
    bencher.bench_local(|| {
        mm_ref(&w, &x, &mut y, m, 1, k);
        std::hint::black_box(&y);
    });
}

#[divan::bench(args = GEMM_SHAPES)]
fn mm_gemm(bencher: Bencher, &(m, n, k): &(usize, usize, usize)) {
    let (a, b, mut c) = setup_gemm(m, n, k);
    bencher.bench_local(|| {
        mm_ref(&a, &b, &mut c, m, n, k);
        std::hint::black_box(&c);
    });
}

#[divan::bench(args = GEMM_LM_HEAD, sample_count = 3, sample_size = 1)]
fn mm_gemm_lm(bencher: Bencher, &(m, n, k): &(usize, usize, usize)) {
    let (a, b, mut c) = setup_gemm(m, n, k);
    bencher.bench_local(|| {
        mm_ref(&a, &b, &mut c, m, n, k);
        std::hint::black_box(&c);
    });
}

// ============================================================
// Proper AVX2/FMA GEMV reference (honest CPU baseline)
// ============================================================
//
// matrixmultiply::sgemm with N=1 repacks its B operand every call, which is
// correct but wasteful for GEMV. This reference is what a purpose-built CPU
// GEMV achieves — the number llama.cpp's CPU backend is compared against.

#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_avx2(w: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    use std::arch::x86_64::*;
    for i in 0..m {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= k {
            // SAFETY: `j + 8 <= k` bounds-checks the 8-lane loads against both
            // `row` and `x`, which have length exactly k. Loads are unaligned.
            unsafe {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(row.as_ptr().add(j)),
                    _mm256_loadu_ps(x.as_ptr().add(j)),
                    acc,
                );
            }
            j += 8;
        }
        let mut buf = [0.0f32; 8];
        // SAFETY: buf is exactly 8 f32s (32 bytes); _mm256_storeu_ps writes
        // exactly 32 bytes to a properly aligned-for-u8 destination.
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
        let mut sum: f32 = buf.iter().sum();
        while j < k {
            sum += row[j] * x[j];
            j += 1;
        }
        y[i] = sum;
    }
}

#[divan::bench(args = GEMV_SHAPES)]
fn gemv_avx2_ref(bencher: Bencher, &(m, k): &(usize, usize)) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("gemv_avx2_ref requires AVX2 and FMA");
    }
    let (w, x, mut y) = setup_gemv(m, k);
    bencher.bench_local(|| {
        // SAFETY: we've just checked AVX2 and FMA at runtime, so
        // `#[target_feature(enable = "avx2,fma")]` on `gemv_avx2` is satisfied.
        unsafe {
            gemv_avx2(&w, &x, &mut y, m, k);
        }
        std::hint::black_box(&y);
    });
}

#[divan::bench(args = GEMV_LM_HEAD, sample_count = 3, sample_size = 1)]
fn gemv_avx2_ref_lm(bencher: Bencher, &(m, k): &(usize, usize)) {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("gemv_avx2_ref_lm requires AVX2 and FMA");
    }
    let (w, x, mut y) = setup_gemv(m, k);
    bencher.bench_local(|| {
        // SAFETY: see gemv_avx2_ref.
        unsafe {
            gemv_avx2(&w, &x, &mut y, m, k);
        }
        std::hint::black_box(&y);
    });
}
