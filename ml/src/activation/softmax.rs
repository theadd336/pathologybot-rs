use ndarray::{Dimension, NdFloat};

use super::ActivationFunction;

pub struct SoftmaxActivation;

impl<A: NdFloat, D: Dimension> ActivationFunction<A, D> for SoftmaxActivation {
    fn compute(&self, inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        todo!()
    }

    fn compute_derivative(&self, inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        todo!()
    }
}
