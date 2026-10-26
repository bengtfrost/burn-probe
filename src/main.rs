// src/main.rs
#![recursion_limit = "256"]

use burn::tensor::{Distribution, Tensor};
use burn_wgpu::Vulkan;

type B = Vulkan<f32, i32>;

fn main() {
    let device = Default::default(); // burn_wgpu default device targets Vulkan
    let a: Tensor<B, 2> = Tensor::random([64, 64], Distribution::Default, &device);
    let b: Tensor<B, 2> = Tensor::random([64, 64], Distribution::Default, &device);
    let c = a.matmul(b);
    println!("shape: {:?}", c.shape());
    println!("sum:   {}", c.sum().into_scalar());
}
