use ndarray::{ArcArray, Array, NdFloat};

use crate::optimizers::Optimizer;

pub mod dense;

pub trait Layer<A, O> {
    type InputDim;
    type OutputDim;
    fn compute(&mut self, input: ArcArray<A, Self::InputDim>) -> ArcArray<A, Self::OutputDim>;
    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, Self::InputDim>,
        prior_errors: Array<A, Self::OutputDim>,
    ) -> Array<A, Self::InputDim>;
}

pub trait LayerBuilder<A: NdFloat, O: Optimizer<A>> {
    type LayerImpl: Layer<A, O>;
    fn build(self, optimizer: O, batch_size: A, input_size: usize) -> Self::LayerImpl;
}
