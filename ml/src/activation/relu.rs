use ndarray::Dimension;

use crate::shared::MlNumber;

use super::ActivationFunction;

pub struct ReLUActivation<A> {
    pivot: A,
}

impl<A: MlNumber> ReLUActivation<A> {
    pub fn new() -> Self {
        Self { pivot: A::zero() }
    }
}

impl<A: MlNumber> Default for ReLUActivation<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: MlNumber, D: Dimension> ActivationFunction<A, D> for ReLUActivation<A> {
    fn compute(&self, inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        inputs
            .map(|&element| {
                if element >= self.pivot {
                    return element;
                }
                self.pivot
            })
            .to_shared()
    }

    fn compute_derivative(&self, inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        inputs
            .map(|&element| {
                if element >= self.pivot {
                    return A::one();
                }
                A::zero()
            })
            .to_shared()
    }
}
