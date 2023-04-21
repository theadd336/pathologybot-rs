use conv::ValueInto;
use ndarray::{ArcArray, IxDyn};

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

    pub fn train(&mut self, training_data: ArcArray<A, IxDyn>) {
        let mut next_input = training_data;
        for layer in self.layers.iter_mut() {
            let output = layer.compute(next_input);
            next_input = output;
        }
    }
}
