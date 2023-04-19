use ndarray::{ArcArray, Array};

pub mod dense;

pub trait Layer<A> {
    type InputDim;
    type OutputDim;
    fn compute(&mut self, input: ArcArray<A, Self::InputDim>) -> ArcArray<A, Self::OutputDim>;
    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, Self::InputDim>,
        prior_errors: Array<A, Self::OutputDim>,
    ) -> Array<A, Self::InputDim>;
}
