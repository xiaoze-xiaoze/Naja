use std::marker::PhantomData;

use ndarray::{Array1, Axis};
use crate::core::compute::types::{Matrix, MatrixView};
use crate::core::traits::{InverseTransformer, Transformer};
use crate::core::Result;
use super::{Unfitted, Fitted};

#[derive(Debug, Clone)]
pub struct StandardScaler<S = Unfitted> {
    mean: Option<Array1<f64>>,
    std: Option<Array1<f64>>,
    _marker: PhantomData<S>,
}

impl StandardScaler<Unfitted> {
    pub fn new() -> Self {
        Self { mean: None, std: None, _marker: PhantomData }
    }

    pub fn fit(self, data: MatrixView<'_>) -> StandardScaler<Fitted> {
        let mean = data.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(data.ncols()));
        let mut std = data.std_axis(Axis(0), 0.0);
        std.mapv_inplace(|x| if x == 0.0 { 1.0 } else { x });
        StandardScaler { mean: Some(mean), std: Some(std), _marker: PhantomData }
    }
}

impl Default for StandardScaler<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl Transformer for StandardScaler<Fitted> {
    fn transform(&self, data: MatrixView<'_>) -> Result<Matrix> {
        let mut out = data.to_owned();
        out -= self.mean.as_ref().unwrap();
        out /= self.std.as_ref().unwrap();
        Ok(out)
    }
}

impl InverseTransformer for StandardScaler<Fitted> {
    fn inverse_transform(&self, data: MatrixView<'_>) -> Result<Matrix> {
        let mut out = data.to_owned();
        out *= self.std.as_ref().unwrap();
        out += self.mean.as_ref().unwrap();
        Ok(out)
    }
}
