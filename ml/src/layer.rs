use ndarray::{ArcArray, Array, IxDyn};

use crate::{optimizers::Optimizer, shared::MlNumber};

pub mod dense;

pub trait Layer<A, O> {
    fn output_shape(&self) -> &[usize];
    fn compute(&mut self, input: ArcArray<A, IxDyn>) -> ArcArray<A, IxDyn>;
    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, IxDyn>,
        prior_errors: Array<A, IxDyn>,
    ) -> Array<A, IxDyn>;
}

pub trait LayerBuilder<A: MlNumber, O: Optimizer<A>> {
    fn build(self, optimizer: O, batch_size: A, input_size: usize) -> Box<dyn Layer<A, O>>;
}
