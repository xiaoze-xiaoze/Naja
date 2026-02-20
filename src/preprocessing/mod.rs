pub mod scaler;

use crate::core::compute::types::Matrix;
use ndarray::Array1;

pub use scaler::StandardScaler;

pub struct Unfitted;

pub struct Fitted {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

pub trait Transformer {
    fn transform(&self, data: &Matrix) -> Matrix;
}

pub trait Fittable<T> {
    fn fit(self, data: &Matrix) -> T;
}

impl StandardScaler<Unfitted> {
    pub fn fit_transform(self, data: &Matrix) -> (StandardScaler<Fitted>, Matrix) {
        let fitted = self.fit(data);
        let transformed = fitted.transform(data);
        (fitted, transformed)
    }
}
