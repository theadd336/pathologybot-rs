use conv::{ValueFrom, ValueInto};
use ndarray::NdFloat;
use ndarray_rand::rand_distr::uniform::SampleUniform;

pub trait MlNumber: NdFloat + SampleUniform + ValueFrom<usize> {}

impl MlNumber for f32 {}
impl MlNumber for f64 {}
