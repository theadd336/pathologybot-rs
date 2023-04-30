use ndarray::{Array, Dimension, NdFloat};

use super::Optimizer;

#[derive(Debug, Clone, Copy)]
pub struct SGD<A> {
    learning_rate: A,
}

impl<A> SGD<A> {
    pub fn new(learning_rate: A) -> Self {
        Self { learning_rate }
    }
}

impl<A: NdFloat> Optimizer<A> for SGD<A> {
    fn optimize<D: Dimension>(&mut self, gradient: Array<A, D>) -> Array<A, D> {
        gradient * self.learning_rate
    }
}
