use ndarray::{Array, Dimension, NdFloat};

pub mod sgd;

pub trait Optimizer<A: NdFloat>: Clone {
    fn optimize<D: Dimension>(&mut self, gradient: Array<A, D>) -> Array<A, D>;
}
