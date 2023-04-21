use ndarray::{ArcArray, Array, IxDyn};

use crate::shared::MlNumber;

pub trait Loss<A: MlNumber> {
    fn compute_loss(
        computed_output: ArcArray<A, IxDyn>,
        expected_output: ArcArray<A, IxDyn>,
    ) -> Array<A, IxDyn>;
}
