use ndarray::{ArcArray, NdFloat};

pub mod constant;

pub trait ActivationFunction<A: NdFloat, D> {
    fn compute(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
    fn compute_derivative(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
}
