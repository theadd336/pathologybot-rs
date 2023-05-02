use ndarray::{ArcArray, Array, IxDyn};

use crate::shared::MlNumber;

pub mod categorical_cross_entropy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    CategoricalCrossEntropy,
    Custom,
}

pub trait Loss<A: MlNumber> {
    fn compute_loss(
        &self,
        computed_output: ArcArray<A, IxDyn>,
        expected_output: ArcArray<A, IxDyn>,
    ) -> Array<A, IxDyn>;

    fn compute_derivative(
        &self,
        computed_output: ArcArray<A, IxDyn>,
        expected_output: ArcArray<A, IxDyn>,
    ) -> Array<A, IxDyn>;

    fn loss_type(&self) -> LossType {
        LossType::Custom
    }
}
