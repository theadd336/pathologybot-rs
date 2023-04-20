use ndarray::{ArcArray, Array};

use crate::{optimizers::Optimizer, shared::MlNumber};

use self::dense::DenseLayer;

pub mod dense;

pub enum LayerType<A, O> {
    Dense(DenseLayer<A, O>),
}

impl<A: MlNumber, O: Optimizer<A>> LayerType<A, O> {
    pub fn output_shape(&self) -> Vec<usize> {
        match self {
            LayerType::Dense(l) => l.output_shape().into(),
        }
    }
}

pub trait Layer<A, O> {
    type InputDim;
    type OutputDim;

    fn output_shape(&self) -> &[usize];
    fn compute(&mut self, input: ArcArray<A, Self::InputDim>) -> ArcArray<A, Self::OutputDim>;
    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, Self::InputDim>,
        prior_errors: Array<A, Self::OutputDim>,
    ) -> Array<A, Self::InputDim>;
}

pub trait LayerBuilder<A: MlNumber, O: Optimizer<A>> {
    fn build(self, optimizer: O, batch_size: A, input_size: usize) -> LayerType<A, O>;
}
