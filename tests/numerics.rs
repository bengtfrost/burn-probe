use burn::tensor::{Distribution, Tensor};
use burn_wgpu::Vulkan;

type B = Vulkan<f32, i32>;

#[test]
fn burn_vulkan_matches_matrixmultiply() {
    let (m, n, k) = (64, 64, 64);
    let device = Default::default();

    let a: Tensor<B, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random([k, n], Distribution::Default, &device);

    // Pull the exact same inputs back to host so both sides multiply identical data.
    let a_data: Vec<f32> = a.clone().into_data().to_vec().unwrap();
    let b_data: Vec<f32> = b.clone().into_data().to_vec().unwrap();

    let mut c_ref = vec![0.0f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a_data.as_ptr(),
            k as isize,
            1,
            b_data.as_ptr(),
            n as isize,
            1,
            0.0,
            c_ref.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    let c_vec: Vec<f32> = a.matmul(b).into_data().to_vec().unwrap();

    let mut max_rel = 0.0f32;
    for (x, y) in c_vec.iter().zip(c_ref.iter()) {
        let denom = y.abs().max(1e-6);
        max_rel = max_rel.max((x - y).abs() / denom);
    }
    assert!(max_rel < 1e-4, "max rel err = {max_rel}");
}

// ==== f16 GEMV numerics check (added 2026-10-24) ====
//
// Medium GEMV shape. The kernel is f16, the reference is f32 sgemm on the
// quantized-then-upcast inputs. Threshold is `1e-2 * max(|y|, 1)` per
// element — standard f16 GEMV tolerance for k = 2048 with N(0,1) inputs,
// where outputs are ~ N(0, 45). Anything under 1.0 worst/tol is passing.

#[test]
fn burn_vulkan_f16_matches_matrixmultiply() {
    type Bf16 = Vulkan<burn::tensor::f16, i32>;

    let (m, n, k) = (2048, 1, 2048);
    let device = Default::default();

    let a: Tensor<Bf16, 2> = Tensor::random([m, k], Distribution::Default, &device);
    let b: Tensor<Bf16, 2> = Tensor::random([k, n], Distribution::Default, &device);

    let a_f16: Vec<burn::tensor::f16> = a.clone().into_data().to_vec().unwrap();
    let b_f16: Vec<burn::tensor::f16> = b.clone().into_data().to_vec().unwrap();
    let a_data: Vec<f32> = a_f16.into_iter().map(|x| x.to_f32()).collect();
    let b_data: Vec<f32> = b_f16.into_iter().map(|x| x.to_f32()).collect();

    let mut c_ref = vec![0.0f32; m * n];
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a_data.as_ptr(),
            k as isize,
            1,
            b_data.as_ptr(),
            n as isize,
            1,
            0.0,
            c_ref.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    let c_f16: Vec<burn::tensor::f16> = a.matmul(b).into_data().to_vec().unwrap();
    let c_vec: Vec<f32> = c_f16.into_iter().map(|x| x.to_f32()).collect();

    let mut worst = 0.0f32;
    for (x, y) in c_vec.iter().zip(c_ref.iter()) {
        let err = (x - y).abs();
        let tol = 1e-2 * y.abs().max(1.0);
        worst = worst.max(err / tol);
    }
    assert!(
        worst < 1.0,
        "f16 GEMV out of tolerance: worst/tol = {worst}"
    );
}
