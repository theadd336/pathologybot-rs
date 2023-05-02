use ndarray::{ArcArray2, Axis};

use crate::shared::MlNumber;

use super::Loss;

pub struct CategoricalCrossEntropy;

impl<A: MlNumber> Loss<A> for CategoricalCrossEntropy {
    fn compute_loss(
        &self,
        computed_output: ndarray::ArcArray<A, ndarray::IxDyn>,
        expected_output: ndarray::ArcArray<A, ndarray::IxDyn>,
    ) -> ndarray::Array<A, ndarray::IxDyn> {
        let mut computed_output: ArcArray2<A> = computed_output.into_dimensionality().unwrap();
        let expected_output: ArcArray2<A> = expected_output.into_dimensionality().unwrap();
        computed_output.zip_mut_with(&expected_output, |computed_elem, &expected_elem| {
            *computed_elem = -expected_elem * computed_elem.ln();
        });
        computed_output.sum_axis(Axis(1)).into_dyn()
    }

    fn compute_derivative(
        &self,
        _computed_output: ndarray::ArcArray<A, ndarray::IxDyn>,
        _expected_output: ndarray::ArcArray<A, ndarray::IxDyn>,
    ) -> ndarray::Array<A, ndarray::IxDyn> {
        todo!()
    }

    fn loss_type(&self) -> super::LossType {
        super::LossType::CategoricalCrossEntropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_categorical_cross_entropy() {
        let inputs = array![[0.3, 0.7], [0.4, 0.6]];
        let labels = array![[0.0, 1.0], [1.0, 0.0]];
        let output = CategoricalCrossEntropy.compute_loss(
            inputs.into_shared().into_dyn(),
            labels.into_shared().into_dyn(),
        );
        assert_eq!(
            output,
            array![0.35667494393873245, 0.916290731874155].into_dyn()
        );
    }
}
