use std::cell::RefCell;
use std::ops::SubAssign;

use ndarray::{s, ArcArray, ArcArray2, Array, Array1, Array2, Axis, Ix2, NdFloat};
use rand::distributions::uniform::SampleUniform;

use crate::activation::{constant::ConstantActivation, ActivationFunction};
use crate::initializer::{self, initialize_1d, Initializer};

use super::Layer;

pub struct DenseLayer<A> {
    weights: Array2<A>,
    bias: Array1<A>,
    activation: Box<dyn ActivationFunction<A, Ix2>>,
    pre_activation_function_outputs: ArcArray2<A>,
    learning_rate: A,
}

impl<A: NdFloat + SampleUniform> DenseLayer<A> {
    pub(crate) fn new(
        input_size: usize,
        num_nodes: usize,
        activaton_fn: impl ActivationFunction<A, Ix2> + 'static,
        learning_rate: A,
        weight_initializer: Option<Initializer>,
        bias_initializer: Option<Initializer>,
    ) -> Self {
        let weights: Array2<A> = initializer::initialize_2d(
            weight_initializer.unwrap_or(Initializer::Random),
            (num_nodes, input_size),
            None,
            None,
        );
        let bias = initialize_1d(
            bias_initializer.unwrap_or(Initializer::Zeros),
            num_nodes,
            None,
            None,
        );
        Self {
            weights,
            bias,
            learning_rate,
            activation: Box::new(activaton_fn),
            pre_activation_function_outputs: ArcArray2::zeros((0, 0)),
        }
    }
}

impl<A: NdFloat> Layer<A> for DenseLayer<A> {
    type InputDim = Ix2;

    type OutputDim = Ix2;

    fn compute(&mut self, input: ArcArray<A, Self::InputDim>) -> ArcArray<A, Self::OutputDim> {
        let after_weights = self.weights.dot(&input);
        let after_bias = after_weights + &self.bias;
        self.pre_activation_function_outputs = after_bias.into();
        self.activation
            .compute(self.pre_activation_function_outputs.clone())
    }

    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, Self::InputDim>,
        prior_errors: Array<A, Self::OutputDim>,
    ) -> Array<A, Self::InputDim> {
        let derivative = self
            .activation
            .compute_derivative(self.pre_activation_function_outputs.clone());
        let dldb = &prior_errors * derivative;
        let dldw = layer_input.t().dot(&dldb);
        let errors_for_prior_layer = &dldb * &self.weights.t();
        self.weights = (&self.weights - (dldw * self.learning_rate)).to_owned();
        self.bias = &self.bias
            - (dldb.fold_axis(Axis(1), A::zero(), |accum, elem| *accum + *elem)
                * self.learning_rate);
        errors_for_prior_layer.to_owned()
    }
}
