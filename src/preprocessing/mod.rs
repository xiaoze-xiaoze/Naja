pub mod scaler;

use crate::core::compute::types::{Matrix, MatrixView};
use crate::core::traits::Transformer;
use crate::core::Result;

pub use scaler::StandardScaler;

pub struct Unfitted;
pub struct Fitted;

impl StandardScaler<Unfitted> {
    pub fn fit_transform(self, data: MatrixView<'_>) -> Result<(StandardScaler<Fitted>, Matrix)> {
        let fitted = self.fit(data);
        let transformed = fitted.transform(data)?;
        Ok((fitted, transformed))
    }
}
