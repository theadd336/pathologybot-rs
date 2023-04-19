use ndarray::{s, ArcArray, ArcArray2, Array, Array2, Array3, Axis, Ix2, NdFloat};
use rand::distributions::uniform::SampleUniform;

use crate::activation::ActivationFunction;
use crate::initializer::{self, Initializer};

use super::Layer;

pub struct DenseLayer<A> {
    weights: Array2<A>,
    bias: Array2<A>,
    activation: Box<dyn ActivationFunction<A, Ix2>>,
    pre_activation_function_outputs: ArcArray2<A>,
    learning_rate: A,
    #[allow(unused)]
    batch_size: A,
}

impl<A: NdFloat + SampleUniform> DenseLayer<A> {
    pub(crate) fn new(
        input_size: usize,
        num_nodes: usize,
        activaton_fn: impl ActivationFunction<A, Ix2> + 'static,
        learning_rate: A,
        weight_initializer: Option<Initializer>,
        bias_initializer: Option<Initializer>,
        batch_size: A,
    ) -> Self {
        let weights: Array2<A> = initializer::initialize_2d(
            weight_initializer.unwrap_or(Initializer::Random),
            (num_nodes, input_size),
            None,
            None,
        );
        let bias = initializer::initialize_2d(
            bias_initializer.unwrap_or(Initializer::Zeros),
            (num_nodes, 1),
            None,
            None,
        );
        Self {
            weights,
            bias,
            learning_rate,
            batch_size,
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

        let dldb_t = dldb.t();
        assert_eq!(layer_input.ncols(), dldb_t.nrows());

        let mut dldw = Array3::uninit([
            layer_input.ncols(),
            self.weights.nrows(),
            self.weights.ncols(),
        ]);
        for (index, (layer_input_col, dldb_row)) in layer_input
            .columns()
            .into_iter()
            .zip(dldb_t.rows())
            .enumerate()
        {
            let dldb_row = dldb_row.broadcast((1, dldb_row.dim())).unwrap();
            let layer_input_col = layer_input_col.insert_axis(Axis(1));
            let dldw_single_batch = layer_input_col.dot(&dldb_row);
            dldw_single_batch.assign_to(dldw.slice_mut(s![index, .., ..]));
        }
        let dldw = unsafe { dldw.assume_init() };
        let errors_for_prior_layer = self.weights.t().dot(&dldb);
        self.weights =
            &self.weights - (dldw.sum_axis(Axis(0)) * self.learning_rate / self.batch_size);
        self.bias = &self.bias
            - (dldb.sum_axis(Axis(1)).insert_axis(Axis(1)) * self.learning_rate / self.batch_size);
        errors_for_prior_layer.to_owned()
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::activation::relu::ReLUActivation;
    use ndarray::array;

    #[test]
    fn test_dense_layer() {
        let starting_weights = array![[0.3, 0.2], [-0.1, 0.1]];
        let input = array![[0.7], [0.5]];
        let starting_bias = array![[0.2], [-0.1]];

        let mut layer = DenseLayer {
            weights: starting_weights,
            bias: starting_bias,
            activation: Box::new(ReLUActivation::new()),
            pre_activation_function_outputs: ArcArray2::zeros((0, 0)),
            learning_rate: 1.0,
            batch_size: 1.0,
        };
        let output = layer.compute(input.to_shared());
        assert_eq!(output, array![[0.51], [0.0]]);

        let prior_errors = array![[-0.168], [0.056]];
        let backprop_output = layer.backpropogate(input.to_shared(), prior_errors);
        assert_eq!(backprop_output, array![[-0.0504], [-0.033600000000000005]]);
        assert_eq!(
            layer.weights,
            array![[0.41759999999999997, 0.2], [-0.016, 0.1]]
        );
        assert_eq!(layer.bias, array![[0.368], [-0.1]]);
    }
}
