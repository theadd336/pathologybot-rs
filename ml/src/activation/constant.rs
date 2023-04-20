use ndarray::{ArcArray, Dimension};

use crate::shared::MlNumber;

use super::ActivationFunction;

pub struct ConstantActivation;

impl<A: MlNumber, D: Dimension> ActivationFunction<A, D> for ConstantActivation {
    fn compute(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D> {
        inputs
    }
    fn compute_derivative(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D> {
        ArcArray::ones(inputs.raw_dim())
    }
}
