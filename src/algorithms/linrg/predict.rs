use crate::core::{Result, Error};
use crate::core::compute::types::{MatrixView, Vector};
use crate::core::compute::ops;
use crate::core::traits::Predictor;

#[derive(Debug, Clone)]
pub struct LinearRegressionSolution {
    pub coefficients: Vector,
    pub intercept: f64,
}

impl Predictor for LinearRegressionSolution {
    fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        ops::ensure_nonempty_mat(x)?;
        if x.ncols() != self.coefficients.len() { return Err(Error::invalid_shape(format!("Input features mismatch: model has {}, input has {}", self.coefficients.len(), x.ncols()))); }
        let mut y_pred = ops::gemv(x, self.coefficients.view())?;
        if self.intercept != 0.0 { y_pred.mapv_inplace(|v| v + self.intercept); }
        Ok(y_pred)
    }
}