use ndarray::Dimension;

use crate::shared::MlNumber;

use super::ActivationFunction;

pub struct SoftmaxActivation;

impl<A: MlNumber, D: Dimension> ActivationFunction<A, D> for SoftmaxActivation {
    fn compute(&self, _inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        todo!()
    }

    fn compute_derivative(&self, _inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        todo!()
    }
}
