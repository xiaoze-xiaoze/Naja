use std::marker::PhantomData;
use ndarray::Array1;
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::traits::{Component, Fitted, FittableTransformer, Transformer, InversibleTransformer, Unfitted, PartialFit, State};
use crate::core::{Error, Result};

#[derive(Debug, Clone)]
pub struct MinMaxScaler<S: State = Unfitted> {
    min: Option<Vector>,
    scale: Option<Vector>,
    feature_range: (f64, f64),
    _state: PhantomData<S>,
}

impl Component<Unfitted> for MinMaxScaler<Unfitted> {
    type NextState = Fitted;
    type Output = MinMaxScaler<Fitted>;
}

impl Component<Fitted> for MinMaxScaler<Fitted> {
    type NextState = Fitted;
    type Output = Matrix;
}

impl MinMaxScaler<Unfitted> {
    pub fn new() -> Self {
        Self {
            min: None,
            scale: None,
            feature_range: (0.0, 1.0),
            _state: PhantomData,
        }
    }

    pub fn with_feature_range(mut self, min: f64, max: f64) -> Self {
        self.feature_range = (min, max);
        self
    }

    pub fn fit(self, x: MatrixView<'_>) -> Result<MinMaxScaler<Fitted>> {
        ops::ensure_nonempty_mat(x)?;
        let p = x.ncols();
        let mut min = Array1::<f64>::zeros(p);
        let mut max = Array1::<f64>::zeros(p);
        for j in 0..p {
            let col = x.column(j);
            min[j] = col.fold(f64::INFINITY, |a, &b| a.min(b));
            max[j] = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }
        let scale = max - &min;
        Ok(MinMaxScaler {
            min: Some(min),
            scale: Some(scale),
            feature_range: self.feature_range,
            _state: PhantomData,
        })
    }
}

impl Default for MinMaxScaler<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl FittableTransformer for MinMaxScaler<Unfitted> {
    fn fit(self, x: MatrixView<'_>) -> Result<Self::Output> {
        self.fit(x)
    }
}

impl MinMaxScaler<Fitted> {
    pub fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let min = self.min.as_ref().ok_or_else(|| Error::invalid_state("min not computed"))?;
        let scale = self.scale.as_ref().ok_or_else(|| Error::invalid_state("scale not computed"))?;
        if x.ncols() != min.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), min.len())));
        }
        let (min_range, max_range) = self.feature_range;
        let range = max_range - min_range;
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let mn = min[j];
            let sc = if scale[j] < 1e-8 { 1.0 } else { scale[j] };
            out.column_mut(j).mapv_inplace(|v| (v - mn) / sc * range + min_range);
        }
        Ok(out)
    }

    pub fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let min = self.min.as_ref().ok_or_else(|| Error::invalid_state("min not computed"))?;
        let scale = self.scale.as_ref().ok_or_else(|| Error::invalid_state("scale not computed"))?;
        if x.ncols() != min.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), min.len())));
        }
        let (min_range, max_range) = self.feature_range;
        let range = max_range - min_range;
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let mn = min[j];
            let sc = if scale[j] < 1e-8 { 1.0 } else { scale[j] };
            out.column_mut(j).mapv_inplace(|v| (v - min_range) / range * sc + mn);
        }
        Ok(out)
    }
}

impl Transformer for MinMaxScaler<Fitted> {
    fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.transform(x)
    }
}

impl InversibleTransformer for MinMaxScaler<Fitted> {
    fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.inverse_transform(x)
    }
}

impl PartialFit for MinMaxScaler<Fitted> {
    fn partial_fit(&mut self, x: MatrixView<'_>, _: Option<VectorView<'_>>) -> Result<()> {
        ops::ensure_nonempty_mat(x)?;
        let p = x.ncols();
        let min = self.min.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        let scale = self.scale.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        if min.len() != p {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", p, min.len())));
        }
        for j in 0..p {
            let col = x.column(j);
            let batch_min = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let batch_max = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if batch_min < min[j] { min[j] = batch_min; }
            let old_max = min[j] + scale[j];
            if batch_max > old_max { scale[j] = batch_max - min[j]; }
        }
        Ok(())
    }
}
