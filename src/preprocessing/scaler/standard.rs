use std::marker::PhantomData;
use ndarray::{Array1, Axis};
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::traits::{Component, Fitted, FittableTransformer, Transformer, InversibleTransformer, Unfitted, PartialFit, State};
use crate::core::{Error, Result};

#[derive(Debug, Clone)]
pub struct StandardScaler<S: State = Unfitted> {
    mean: Option<Vector>,
    std: Option<Vector>,
    n: usize,
    sum: Option<Vector>,
    sum_sq: Option<Vector>,
    _state: PhantomData<S>,
}

impl Component<Unfitted> for StandardScaler<Unfitted> {
    type NextState = Fitted;
    type Output = StandardScaler<Fitted>;
}

impl Component<Fitted> for StandardScaler<Fitted> {
    type NextState = Fitted;
    type Output = Matrix;
}

impl StandardScaler<Unfitted> {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            n: 0,
            sum: None,
            sum_sq: None,
            _state: PhantomData,
        }
    }

    pub fn fit(self, x: MatrixView<'_>) -> Result<StandardScaler<Fitted>> {
        ops::ensure_nonempty_mat(x)?;
        let n = x.nrows();
        let p = x.ncols();
        let mut mean = Array1::<f64>::zeros(p);
        let mut std = Array1::<f64>::zeros(p);
        for j in 0..p {
            let col = x.column(j);
            mean[j] = col.mean().unwrap_or(0.0);
            let var = col.mapv(|v| (v - mean[j]).powi(2)).mean().unwrap_or(0.0);
            std[j] = var.sqrt();
        }
        Ok(StandardScaler {
            mean: Some(mean),
            std: Some(std),
            n,
            sum: Some(x.sum_axis(Axis(0))),
            sum_sq: Some(x.mapv(|v| v * v).sum_axis(Axis(0))),
            _state: PhantomData,
        })
    }
}

impl Default for StandardScaler<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl FittableTransformer for StandardScaler<Unfitted> {
    fn fit(self, x: MatrixView<'_>) -> Result<Self::Output> {
        self.fit(x)
    }
}

impl StandardScaler<Fitted> {
    pub fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let mean = self.mean.as_ref().ok_or_else(|| Error::invalid_state("mean not computed"))?;
        let std = self.std.as_ref().ok_or_else(|| Error::invalid_state("std not computed"))?;
        if x.ncols() != mean.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), mean.len())));
        }
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let mj = mean[j];
            let sj = if std[j] < 1e-8 { 1.0 } else { std[j] };
            out.column_mut(j).mapv_inplace(|v| (v - mj) / sj);
        }
        Ok(out)
    }

    pub fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        ops::ensure_nonempty_mat(x)?;
        let mean = self.mean.as_ref().ok_or_else(|| Error::invalid_state("mean not computed"))?;
        let std = self.std.as_ref().ok_or_else(|| Error::invalid_state("std not computed"))?;
        if x.ncols() != mean.len() {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", x.ncols(), mean.len())));
        }
        let mut out = x.to_owned();
        for j in 0..x.ncols() {
            let mj = mean[j];
            let sj = if std[j] < 1e-8 { 1.0 } else { std[j] };
            out.column_mut(j).mapv_inplace(|v| v * sj + mj);
        }
        Ok(out)
    }
}

impl Transformer for StandardScaler<Fitted> {
    fn transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.transform(x)
    }
}

impl InversibleTransformer for StandardScaler<Fitted> {
    fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.inverse_transform(x)
    }
}

impl PartialFit for StandardScaler<Fitted> {
    fn partial_fit(&mut self, x: MatrixView<'_>, _: Option<VectorView<'_>>) -> Result<()> {
        ops::ensure_nonempty_mat(x)?;
        let m = x.nrows();
        let p = x.ncols();
        let mean = self.mean.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        let std = self.std.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        let sum = self.sum.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        let sum_sq = self.sum_sq.as_mut().ok_or_else(|| Error::invalid_state("not fitted"))?;
        if mean.len() != p {
            return Err(Error::invalid_shape(format!("input has {} columns, expected {}", p, mean.len())));
        }
        let n_old = self.n as f64;
        let n_new = (self.n + m) as f64;
        let batch_sum = x.sum_axis(Axis(0));
        let batch_sum_sq = x.mapv(|v| v * v).sum_axis(Axis(0));
        let old_mean = mean.clone();
        *mean = (&old_mean * n_old + &batch_sum) / n_new;
        *sum = &*sum + &batch_sum;
        *sum_sq = &*sum_sq + &batch_sum_sq;
        for j in 0..p {
            let total_ss = sum_sq[j] - n_new * mean[j] * mean[j];
            std[j] = (total_ss / n_new).max(0.0).sqrt();
        }
        self.n += m;
        Ok(())
    }
}
