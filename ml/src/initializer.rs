use ndarray::{Array1, Array2, NdFloat};
use ndarray_rand::RandomExt;
use rand::distributions::{uniform::SampleUniform, Uniform};
use rand::prelude::Distribution;

pub enum Initializer {
    Zeros,
    Ones,
    Random,
}

pub fn initialize_1d<A: NdFloat + SampleUniform>(
    initializer: Initializer,
    len: usize,
    start: Option<A>,
    end: Option<A>,
) -> Array1<A> {
    let start = start.unwrap_or_else(|| A::neg(A::one()));
    let end = end.unwrap_or_else(A::one);
    match initializer {
        Initializer::Zeros => Array1::zeros(len),
        Initializer::Ones => Array1::ones(len),
        Initializer::Random => {
            let mut rng = rand::thread_rng();
            let uniform_sample = Uniform::from(start..end);
            Array1::from_shape_simple_fn(len, || uniform_sample.sample(&mut rng))
        }
    }
}

pub fn initialize_2d<A: NdFloat + SampleUniform>(
    initializer: Initializer,
    shape: (usize, usize),
    start: Option<A>,
    end: Option<A>,
) -> Array2<A> {
    let start = start.unwrap_or_else(|| A::neg(A::one()));
    let end = end.unwrap_or_else(A::one);
    match initializer {
        Initializer::Zeros => Array2::zeros(shape),
        Initializer::Ones => Array2::ones(shape),
        Initializer::Random => {
            let uniform = ndarray_rand::rand_distr::Uniform::new(start, end);
            Array2::random(shape, uniform)
        }
    }
}
