use conv::ValueInto;
use ndarray::{Array, Axis, IxDyn, Slice};

use crate::{
    layer::{Layer, LayerBuilder},
    loss::Loss,
    optimizers::Optimizer,
    shared::MlNumber,
};

pub struct Model<A, O, L> {
    layers: Vec<Box<dyn Layer<A, O>>>,
    optimizer: O,
    loss: L,
    next_input_shape: Vec<usize>,
    batch_size: A,
}

impl<A: MlNumber, O: Optimizer<A>, L: Loss<A>> Model<A, O, L> {
    pub fn new(input_size: usize, batch_size: usize, optimizer: O, loss: L) -> Self {
        Self {
            optimizer,
            loss,
            layers: Vec::new(),
            next_input_shape: vec![input_size, batch_size],
            batch_size: batch_size.value_into().unwrap(),
        }
    }

    pub fn add_layer(&mut self, layer: impl LayerBuilder<A, O>) {
        let layer_impl = layer.build(
            self.optimizer.clone(),
            self.batch_size,
            self.next_input_shape[0],
        );
        self.next_input_shape = layer_impl.output_shape().into();
        self.layers.push(layer_impl);
    }

    pub fn train(
        &mut self,
        training_data: Array<A, IxDyn>,
        labels: Array<A, IxDyn>,
        epochs: usize,
        batch_size: usize,
    ) {
        let num_batches = training_data.shape()[0] / batch_size;
        let has_remainder = (training_data.shape()[0] % batch_size) != 0;
        let mut layer_inputs = Vec::with_capacity(self.layers.len());
        for epoch in 0..epochs {
            for index in 0..=num_batches {
                let mut next_input;
                let batched_labels;
                if index == num_batches && has_remainder {
                    next_input = training_data
                        .slice_axis(Axis(0), Slice::new((index * batch_size) as isize, None, 1))
                        .to_shared();
                    batched_labels = labels
                        .slice_axis(Axis(0), Slice::new((index * batch_size) as isize, None, 1))
                        .to_shared();
                } else if index == num_batches {
                    break;
                } else {
                    next_input = training_data
                        .slice_axis(
                            Axis(0),
                            Slice::new(
                                (index * batch_size) as isize,
                                Some(((index + 1) * batch_size) as isize),
                                1,
                            ),
                        )
                        .to_shared();
                    batched_labels = labels
                        .slice_axis(
                            Axis(0),
                            Slice::new(
                                (index * batch_size) as isize,
                                Some(((index + 1) * batch_size) as isize),
                                1,
                            ),
                        )
                        .to_shared();
                }
                for layer in self.layers.iter_mut() {
                    let output = layer.compute(next_input.clone());
                    layer_inputs.push(next_input);
                    next_input = output;
                }
                let loss = self
                    .loss
                    .compute_loss(next_input.clone(), batched_labels.clone());
                println!("Epoch: {epoch}, loss: {loss:?}");
                let mut backprop_error = self.loss.compute_derivative(next_input, batched_labels);
                for (layer, layer_input) in self.layers.iter_mut().rev().zip(layer_inputs.drain(..))
                {
                    backprop_error = layer.backpropogate(layer_input, backprop_error);
                }
            }
        }
    }
}
