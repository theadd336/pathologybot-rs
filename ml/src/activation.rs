use ndarray::{ArcArray, Dimension};

use crate::shared::MlNumber;

pub mod constant;
pub mod relu;
pub mod softmax;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    Softmax,
    Other,
}

pub trait ActivationFunction<A: MlNumber, D: Dimension> {
    fn compute(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
    fn compute_derivative(&self, inputs: ArcArray<A, D>) -> ArcArray<A, D>;
    fn activation_type(&self) -> ActivationType {
        ActivationType::Other
    }
}
