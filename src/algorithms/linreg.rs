use ndarray::{Array1, Array2};
use crate::core::{Result, Error};
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};
use crate::core::compute::ops;

pub struct LinearRegression {}

impl LinearRegression {
    pub fn new() -> Self { Self {} }

    pub fn configure(self, args:ConfigArgs) -> LinearRegressionSpec { LinearRegressionSpec { config: args } }
}

impl Default for LinearRegression { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Penalty {
    None,
    Ridge { alpha: f64 }
}

#[derive(Debug, Clone)]
pub struct ConfigArgs {
    pub intercept: bool,
    pub penalty: Penalty
}

impl Default for ConfigArgs { fn default() -> Self { Self { intercept: true, penalty: Penalty::None } } }

pub struct LinearRegressionSpec { config: ConfigArgs }

impl LinearRegressionSpec {
    pub fn solve(self, args: SolveArgs<'_>) -> Result<LinearRegressionSolution> {
        ops::ensure_nonempty_mat(args.x)?;
        ops::ensure_nonempty_vec(args.y)?;
        ops::ensure_len(args.y.view(), args.x.column(0).view(), "y", "x(rows)")?;
        let x_design = if self.config.intercept { ops::add_intercept(args.x.clone())? } else { args.x.clone() };
        let mut xtx = ops::xtx(x_design.view())?;
        match self.config.penalty {
            Penalty::None => {}
            Penalty::Ridge { alpha } => {
                let start_idx = if self.config.intercept { 1 } else { 0 };
                for i in start_idx..xtx.nrows() { xtx[[i, i]] += alpha; }
            }
        }
        let xty = ops::xty(x_design.view(), args.y.view())?;
        let w = ops::solve_cholesky(xtx.view(), xty.view()).map_err(|e| Error::lin_alg(format!("Failed to solve normal equation: {}", e)))?;
        let (intercept, coefficients) = if self.config.intercept {
            let intercept = w[0];
            let coeffs = w.slice(ndarray::s![1..]).to_owned();
            (intercept, coeffs);
        } else { (0.0, w) };
        Ok(LinearRegressionSolution { coefficients, intercept })
    }
}

pub struct SolveArgs<'a> {
    pub x: MatrixView<'a>,
    pub y: VectorView<'a>
}

#[derive(Debug, Clone)]
pub struct LinearRegressionSolution {
    pub coefficients: Vector,
    pub intercept: f64,
}

impl LinearRegressionSolution {
    pub fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        ops::ensure_nonempty_mat(x)?;
        if x.ncols() != self.coefficients.len() { return Err(Error::invalid_shape(format!("Input features mismatch: model has {}, input has {}", self.coefficients.len(), x.ncols()))); }
        let mut y_pred = ops::gemv(x, self.coefficients.view())?;
        if self.intercept != 0.0 { y_pred.mapv_inplace(|v| v + self.intercept); }
        Ok(y_pred)
    }
}