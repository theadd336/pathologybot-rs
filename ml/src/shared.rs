use ndarray::NdFloat;
use ndarray_rand::rand_distr::uniform::SampleUniform;

pub trait MlNumber: NdFloat + SampleUniform {}

impl MlNumber for f32 {}
impl MlNumber for f64 {}
