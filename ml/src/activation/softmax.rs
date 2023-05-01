use ndarray::{Axis, Dimension, RemoveAxis};

use crate::shared::MlNumber;

use super::ActivationFunction;

pub struct SoftmaxActivation {
    axis: Option<Axis>,
}

impl SoftmaxActivation {
    pub fn new() -> Self {
        Self { axis: None }
    }
}

impl<A: MlNumber, D: Dimension + RemoveAxis> ActivationFunction<A, D> for SoftmaxActivation {
    fn compute(&self, inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        let axis = self.axis.unwrap_or(Axis(inputs.shape().len() - 1));
        let outputs = inputs.map(|elem| elem.exp());
        let sums = outputs.sum_axis(axis);
        (outputs / sums).to_shared()
    }

    fn compute_derivative(&self, _inputs: ndarray::ArcArray<A, D>) -> ndarray::ArcArray<A, D> {
        todo!();
    }

    fn activation_type(&self) -> super::ActivationType {
        super::ActivationType::Softmax
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::activation::ActivationFunction;

    use super::SoftmaxActivation;

    #[test]
    fn test_softmax_compute() {
        let softmax = SoftmaxActivation::new();
        let inputs = array![[1.0, 2.0, 8.0]];
        let output = softmax.compute(inputs.to_shared());
        assert_eq!(
            output,
            array![[
                0.0009088005553630329,
                0.002470376035336821,
                0.9966208234093001
            ]]
        );
    }
}
