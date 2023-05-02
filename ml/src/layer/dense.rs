use ndarray::{s, ArcArray, ArcArray2, Array, Array2, Array3, ArrayView2, Axis, Ix2, IxDyn};

use crate::activation::constant::ConstantActivation;
use crate::activation::ActivationFunction;
use crate::initializer::{self, Initializer};
use crate::optimizers::Optimizer;
use crate::shared::MlNumber;

use super::{Layer, LayerBuilder};

pub struct DenseLayerBuilder<A> {
    #[allow(unused)]
    starting_weights: Option<Array2<A>>,
    #[allow(unused)]
    starting_bias: Option<Array2<A>>,
    activation: Option<Box<dyn ActivationFunction<A, Ix2>>>,
    weight_initializer: Option<Initializer>,
    bias_initializer: Option<Initializer>,
    num_nodes: Option<usize>,
}

impl<A: MlNumber> DenseLayerBuilder<A> {
    pub fn new() -> Self {
        Self {
            starting_weights: None,
            starting_bias: None,
            activation: Some(Box::new(ConstantActivation)),
            weight_initializer: Some(Initializer::Random),
            bias_initializer: Some(Initializer::Zeros),
            num_nodes: None,
        }
    }
}

impl<A: MlNumber, O: Optimizer<A> + 'static> LayerBuilder<A, O> for DenseLayerBuilder<A> {
    fn build(self, optimizer: O, input_size: usize) -> Box<dyn Layer<A, O>> {
        Box::new(DenseLayer::new(
            input_size,
            self.num_nodes.unwrap(),
            self.activation.unwrap(),
            self.weight_initializer,
            self.bias_initializer,
            optimizer,
        ))
    }
}

pub struct DenseLayer<A, O> {
    weights: Array2<A>,
    bias: Array2<A>,
    activation: Box<dyn ActivationFunction<A, Ix2>>,
    pre_activation_function_outputs: ArcArray2<A>,
    weight_optimizer: O,
    bias_optimizer: O,
}

impl<A: MlNumber, O: Optimizer<A>> DenseLayer<A, O> {
    pub(crate) fn new(
        input_size: usize,
        num_nodes: usize,
        activation: Box<dyn ActivationFunction<A, Ix2>>,
        weight_initializer: Option<Initializer>,
        bias_initializer: Option<Initializer>,
        optimizer: O,
    ) -> Self {
        let weights: Array2<A> = initializer::initialize_2d(
            weight_initializer.unwrap_or(Initializer::Random),
            (input_size, num_nodes),
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
            activation,
            pre_activation_function_outputs: ArcArray2::zeros((0, 0)),
            weight_optimizer: optimizer.clone(),
            bias_optimizer: optimizer,
        }
    }

    fn create_dldw_array(&self, layer_input_t: ArrayView2<A>, sigma: ArcArray2<A>) -> Array3<A> {
        let mut dldw = Array3::uninit([
            layer_input_t.ncols(),
            self.weights.nrows(),
            self.weights.ncols(),
        ]);
        for (index, (layer_input_col, dldb_row)) in layer_input_t
            .columns()
            .into_iter()
            .zip(sigma.rows())
            .enumerate()
        {
            let dldb_row = dldb_row.broadcast((1, dldb_row.dim())).unwrap();
            let layer_input_col = layer_input_col.insert_axis(Axis(1));
            let dldw_single_batch = layer_input_col.dot(&dldb_row);
            dldw_single_batch.assign_to(dldw.slice_mut(s![index, .., ..]));
        }
        unsafe { dldw.assume_init() }
    }
}

impl<A: MlNumber, O: Optimizer<A>> Layer<A, O> for DenseLayer<A, O> {
    fn output_shape(&self) -> &[usize] {
        &self.bias.shape()
    }

    fn compute(&mut self, input: ArcArray<A, IxDyn>) -> ArcArray<A, IxDyn> {
        let input: ArcArray2<A> = input.into_dimensionality().unwrap();
        let after_weights = input.dot(&self.weights);
        let after_bias = after_weights + &self.bias;
        self.pre_activation_function_outputs = after_bias.into();
        (self
            .activation
            .compute(self.pre_activation_function_outputs.clone()))
        .into_dimensionality()
        .unwrap()
    }

    fn backpropogate(
        &mut self,
        layer_input: ArcArray<A, IxDyn>,
        prior_errors: Array<A, IxDyn>,
    ) -> Array<A, IxDyn> {
        let layer_input: ArcArray2<A> = layer_input.into_dimensionality().unwrap();
        let batch_size: A = A::value_from(layer_input.nrows()).unwrap();
        let prior_errors: Array2<A> = prior_errors.into_dimensionality().unwrap();
        let derivative = self
            .activation
            .compute_derivative(self.pre_activation_function_outputs.clone());
        let dldb = &prior_errors * derivative;
        let layer_input_t = layer_input.t();
        assert_eq!(layer_input_t.ncols(), dldb.nrows());
        let dldw = self.create_dldw_array(layer_input_t, dldb.clone());
        let errors_for_prior_layer = dldb.dot(&self.weights.t());

        self.weights = &self.weights
            - self
                .weight_optimizer
                .optimize(dldw.sum_axis(Axis(0)) / batch_size);
        self.bias = &self.bias
            - self
                .bias_optimizer
                .optimize(dldb.sum_axis(Axis(0)).insert_axis(Axis(0)) / batch_size);
        errors_for_prior_layer.to_owned().into_dyn()
    }

    fn backpropogate_specialized(
        &mut self,
        layer_input: ArcArray<A, IxDyn>,
        layer_output: ArcArray<A, IxDyn>,
        labels: ArcArray<A, IxDyn>,
        loss_type: crate::loss::LossType,
    ) -> Array<A, IxDyn> {
        if loss_type != crate::loss::LossType::CategoricalCrossEntropy
            || self.activation.activation_type() != crate::activation::ActivationType::Softmax
        {
            panic!("Specialization is not valid here");
        }
        let layer_input: ArcArray2<A> = layer_input.into_dimensionality().unwrap();
        let batch_size: A = A::value_from(layer_input.nrows()).unwrap();
        let labels: ArcArray2<A> = labels.into_dimensionality().unwrap();
        let layer_output: ArcArray2<A> = layer_output.into_dimensionality().unwrap();
        let derivative = layer_output - labels;
        let errors_for_prior_layer = derivative.dot(&self.weights.t()).into_dyn();
        let dldw = self.create_dldw_array(layer_input.t(), derivative.clone());
        self.weights = &self.weights
            - self
                .weight_optimizer
                .optimize(dldw.sum_axis(Axis(0)) / batch_size);
        self.bias = &self.bias
            - self
                .bias_optimizer
                .optimize(derivative.sum_axis(Axis(0)).insert_axis(Axis(0)) / batch_size);
        errors_for_prior_layer
    }

    fn has_specialization(&self, loss_type: crate::loss::LossType) -> bool {
        loss_type == crate::loss::LossType::CategoricalCrossEntropy
            && self.activation.activation_type() == crate::activation::ActivationType::Softmax
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::{
        activation::{relu::ReLUActivation, softmax::SoftmaxActivation},
        optimizers::sgd::SGD,
    };
    use ndarray::array;

    #[test]
    fn test_dense_layer_normal_functions() {
        let starting_weights = array![[0.3, -0.1], [0.2, 0.1]];
        let input = array![[0.7, 0.5], [0.7, 0.5]];
        let starting_bias = array![[0.2, -0.1]];

        let mut layer = DenseLayer {
            weights: starting_weights,
            bias: starting_bias,
            activation: Box::new(ReLUActivation::new()),
            pre_activation_function_outputs: ArcArray2::zeros((0, 0)),
            bias_optimizer: SGD::new(1.0),
            weight_optimizer: SGD::new(1.0),
        };
        let output = layer.compute(input.to_shared().into_dyn());
        assert_eq!(output, array![[0.51, 0.0], [0.51, 0.0]].into_dyn());

        let prior_errors = array![[-0.168, 0.056], [-0.168, 0.056]];
        let backprop_output =
            layer.backpropogate(input.to_shared().into_dyn(), prior_errors.into_dyn());
        assert_eq!(
            backprop_output,
            array![
                [-0.0504, -0.033600000000000005],
                [-0.0504, -0.033600000000000005]
            ]
            .into_dyn()
        );
        assert_eq!(
            layer.weights,
            array![[0.41759999999999997, -0.1], [0.28400000000000003, 0.1]]
        );
        assert_eq!(layer.bias, array![[0.368, -0.1]]);
    }

    #[test]
    fn test_dense_layer_cross_entropy_loss_softmax_specialization() {
        let starting_weights = array![[0.1, -0.2], [0.1, 0.2]];
        let starting_bias = array![[0.1, 0.0]];
        let layer_inputs = array![[0.51, 0.0], [0.51, 0.0]];
        let layer_outputs = array![[0.56, 0.44], [0.56, 0.44]];
        let mut layer = DenseLayer {
            weights: starting_weights,
            bias: starting_bias,
            activation: Box::new(SoftmaxActivation::new()),
            pre_activation_function_outputs: ArcArray2::zeros((0, 0)),
            bias_optimizer: SGD::new(1.0),
            weight_optimizer: SGD::new(1.0),
        };

        let labels = array![[0.0, 1.0], [0.0, 1.0]];
        assert!(layer.has_specialization(crate::loss::LossType::CategoricalCrossEntropy));
        let errors_for_prior_layer = layer.backpropogate_specialized(
            layer_inputs.into_shared().into_dyn(),
            layer_outputs.into_shared().into_dyn(),
            labels.into_shared().into_dyn(),
            crate::loss::LossType::CategoricalCrossEntropy,
        );

        assert_eq!(
            errors_for_prior_layer,
            array![
                [0.16800000000000004, -0.05600000000000001],
                [0.16800000000000004, -0.05600000000000001]
            ]
            .into_dyn()
        );

        assert_eq!(
            layer.weights,
            array![[-0.18560000000000001, 0.08560000000000001], [0.1, 0.2]]
        );
        assert_eq!(layer.bias, array![[-0.4600000000000001, 0.56]]);
    }
}
