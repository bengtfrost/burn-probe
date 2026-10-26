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
