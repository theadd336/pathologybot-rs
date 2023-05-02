use ndarray::{ArcArray, Array, IxDyn};

use crate::{loss::LossType, optimizers::Optimizer, shared::MlNumber};

pub mod dense;

pub trait Layer<A, O> {
    fn output_shape(&self) -> &[usize];
    fn compute(&mut self, input: ArcArray<A, IxDyn>) -> ArcArray<A, IxDyn>;
    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, IxDyn>,
        prior_errors: Array<A, IxDyn>,
    ) -> Array<A, IxDyn>;
    fn has_specialization(&self, _loss_type: LossType) -> bool {
        false
    }
    #[allow(unused)]
    fn backpropogate_specialized(
        &mut self,
        layer_input: ArcArray<A, IxDyn>,
        layer_output: ArcArray<A, IxDyn>,
        labels: ArcArray<A, IxDyn>,
        loss_type: LossType,
    ) -> Array<A, IxDyn> {
        unimplemented!()
    }
}

pub trait LayerBuilder<A: MlNumber, O: Optimizer<A>> {
    fn build(self, optimizer: O, input_size: usize) -> Box<dyn Layer<A, O>>;
}
