use ndarray::{ArcArray, Dimension, NdFloat};

pub mod constant;
pub mod relu;
pub mod softmax;

pub trait ActivationFunction<A: NdFloat, D: Dimension> {
    fn compute(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
    fn compute_derivative(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
}
