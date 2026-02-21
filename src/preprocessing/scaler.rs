use ndarray::{Array1, Axis};
use crate::core::compute::types::{Matrix, MatrixView};
use crate::core::traits::{InverseTransformer, Transformer};
use crate::core::Result;
use super::{Unfitted, Fitted};

#[derive(Debug, Clone)]
pub struct StandardScaler<S = Unfitted> { state: S }

impl StandardScaler<Unfitted> {
    pub fn new() -> Self { Self { state: Unfitted } }

    pub fn fit(self, data: &Matrix) -> StandardScaler<Fitted> {
        let mean = data.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(data.ncols()));
        let mut std = data.std_axis(Axis(0), 0.0);
        std.mapv_inplace(|x| if x == 0.0 { 1.0 } else { x });
        StandardScaler { state: Fitted { mean, std } }
    }
}

impl Default for StandardScaler<Unfitted> { 
    fn default() -> Self { Self::new() } 
}

impl Transformer for StandardScaler<Fitted> {
    fn transform(&self, data: MatrixView<'_>) -> Result<Matrix> { 
        let mut out = data.to_owned();
        out -= &self.state.mean;
        out /= &self.state.std;
        Ok(out)
    }
}

impl InverseTransformer for StandardScaler<Fitted> {
    fn inverse_transform(&self, data: MatrixView<'_>) -> Result<Matrix> { 
        let mut out = data.to_owned();
        out *= &self.state.std;
        out += &self.state.mean;
        Ok(out)
    }
}
