pub mod scaler;

use crate::core::compute::types::Matrix;
use crate::core::traits::Transformer;
use crate::core::Result;
use ndarray::Array1;

pub use scaler::StandardScaler;

#[derive(Debug, Clone, Copy)]
pub struct Unfitted;

#[derive(Debug, Clone)]
pub struct Fitted {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

pub trait Fittable<T> {
    fn fit(self, data: &Matrix) -> T;
}

impl StandardScaler<Unfitted> {
    pub fn fit_transform(self, data: &Matrix) -> Result<(StandardScaler<Fitted>, Matrix)> {
        let fitted = self.fit(data);
        let transformed = fitted.transform(data.view())?;
        Ok((fitted, transformed))
    }
}
