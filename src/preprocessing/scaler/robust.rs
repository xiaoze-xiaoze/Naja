use std::marker::PhantomData;
use ndarray::Array1;
use crate::core::compute::types::{Matrix, MatrixView, Vector};
use crate::core::compute::ops;
use crate::core::traits::{Component, Fitted, FittableTransformer, Transformer, InversibleTransformer, Unfitted, State};
use crate::core::{Error, Result};
use super::column_quantile;

#[derive(Debug, Clone)]
pub struct RobustScaler<S: State = Unfitted> {
    median: Option<Vector>,
    iqr: Option<Vector>,
    quantile_range: (f64, f64),
    center: bool,
    scale: bool,
    _state: PhantomData<S>,
}

impl Component<Unfitted> for RobustScaler<Unfitted> {
    type NextState = Fitted;
    type Output = RobustScaler<Fitted>;
}

impl Component<Fitted> for RobustScaler<Fitted> {
    type NextState = Fitted;
    type Output = Matrix;
}

impl RobustScaler<Unfitted> {
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            quantile_range: (0.25, 0.75),
            center: true,
            scale: true,
            _state: PhantomData,
        }
    }

    pub fn with_quantile_range(mut self, q1: f64, q3: f64) -> Self {
        self.quantile_range = (q1, q3);
        self
    }

    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    pub fn fit(self, x: MatrixView<'_>) -> Result<RobustScaler<Fitted>> {
        ops::ensure_nonempty_mat(x)?;
        let p = x.ncols();
        let mut median = Array1::<f64>::zeros(p);
        let mut iqr = Array1::<f64>::zeros(p);
        let (q1, q3) = self.quantile_range;
        for j in 0..p {
            let col = x.column(j);
            median[j] = column_quantile(col, 0.5);
            let val_q1 = column_quantile(col, q1);
            let val_q3 = column_quantile(col, q3);
            iqr[j] = val_q3 - val_q1;
        }
        Ok(RobustScaler {
            median: Some(median),
            iqr: Some(iqr),
            quantile_range: self.quantile_range,
            center: self.center,
            scale: self.scale,
            _state: PhantomData,
        })
    }
}

impl Default for RobustScaler<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl FittableTransformer for RobustScaler<Unfitted> {
    fn fit(self, x: MatrixView<'_>) -> Result<Self::Output> {
        self.fit(x)
    }
}

impl RobustScaler<Fitted> {
    pub fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let median = self.median.as_ref().ok_or_else(|| Error::invalid_state("median not computed"))?;
        let iqr = self.iqr.as_ref().ok_or_else(|| Error::invalid_state("iqr not computed"))?;
        if x.ncols() != median.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), median.len())));
        }
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let med = median[j];
            let iq = if iqr[j] < 1e-8 { 1.0 } else { iqr[j] };
            if self.center && self.scale {
                out.column_mut(j).mapv_inplace(|v| (v - med) / iq);
            } else if self.center {
                out.column_mut(j).mapv_inplace(|v| v - med);
            } else if self.scale {
                out.column_mut(j).mapv_inplace(|v| v / iq);
            }
        }
        Ok(out)
    }

    pub fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let median = self.median.as_ref().ok_or_else(|| Error::invalid_state("median not computed"))?;
        let iqr = self.iqr.as_ref().ok_or_else(|| Error::invalid_state("iqr not computed"))?;
        if x.ncols() != median.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), median.len())));
        }
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let med = median[j];
            let iq = if iqr[j] < 1e-8 { 1.0 } else { iqr[j] };
            if self.center && self.scale {
                out.column_mut(j).mapv_inplace(|v| v * iq + med);
            } else if self.center {
                out.column_mut(j).mapv_inplace(|v| v + med);
            } else if self.scale {
                out.column_mut(j).mapv_inplace(|v| v * iq);
            }
        }
        Ok(out)
    }
}

impl Transformer for RobustScaler<Fitted> {
    fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.transform(x)
    }
}

impl InversibleTransformer for RobustScaler<Fitted> {
    fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.inverse_transform(x)
    }
}
