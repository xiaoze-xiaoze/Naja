use ndarray::Array1;
use crate::core::{Result, Error};
use crate::core::compute::types::{MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::regularization::Penalty;
use crate::core::traits::{Predictor, FitSupervised};

use std::fmt;

#[derive(Debug, Clone)]
pub struct LinearRegression {
    intercept: bool,
    penalty: Penalty,
    max_iter: usize,
    tol: f64,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            intercept: true,
            penalty: Penalty::None,
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

impl LinearRegression {
    pub fn new() -> Self { Self::default() }

    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.penalty = penalty;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn solve(self, args: SolveArgs<'_, '_>) -> Result<LinearRegressionSolution> {
        ops::ensure_nonempty_mat(args.x)?;
        ops::ensure_nonempty_vec(args.y)?;
        ops::ensure_len(args.y.view(), args.x.column(0).view(), "y", "x(rows)")?;
        
        let x_design = if self.intercept { ops::add_intercept(args.x.clone())? } else { args.x.to_owned() };
        
        let solution = match self.penalty {
            Penalty::None | Penalty::Ridge { .. } => self.solve_closed_form(x_design.view(), args.y)?,
            Penalty::Lasso { .. } => self.solve_lasso(x_design.view(), args.y)?,
        };

        Ok(solution)
    }

    fn solve_closed_form(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegressionSolution> {
        let mut xtx = ops::xtx(x)?;
        self.penalty.apply_l2(&mut xtx, self.intercept);
        let xty = ops::xty(x, y)?;
        let w = ops::solve_cholesky(xtx.view(), xty.view()).map_err(|e| Error::lin_alg(format!("Failed to solve normal equation: {}", e)))?;
        
        self.package_solution(w)
    }

    fn solve_lasso(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegressionSolution> {
        let n_features = x.ncols();
        let mut w = Array1::<f64>::zeros(n_features);
        let mut norm_sq = Array1::<f64>::zeros(n_features);
        for j in 0..n_features { norm_sq[j] = ops::dot(x.column(j), x.column(j))?; }
        let tol = self.tol;
        let mut r = y.to_owned(); 
        for _iter in 0..self.max_iter {
            let mut max_change: f64 = 0.0;
            for j in 0..n_features {
                let is_intercept = self.intercept && j == 0;
                if norm_sq[j] == 0.0 { continue; }
                let dot_xr = ops::dot(x.column(j), r.view())?;
                let rho = dot_xr + w[j] * norm_sq[j];
                let old_w_j = w[j];
                let new_w_j;
                if is_intercept { new_w_j = rho / norm_sq[j]; } 
                else { new_w_j = self.penalty.apply_l1(rho) / norm_sq[j]; }
                if (new_w_j - old_w_j).abs() > 1e-15 {
                    let diff = new_w_j - old_w_j;
                    ops::add_scaled_mut(&mut r, x.column(j), -diff)?;
                    w[j] = new_w_j;
                    max_change = max_change.max(diff.abs());
                }
            }
            if max_change < tol { break; }
        }
        
        self.package_solution(w)
    }

    fn package_solution(&self, w: Vector) -> Result<LinearRegressionSolution> {
        let (intercept, coefficients) = if self.intercept {
            let intercept = w[0];
            let coeffs = w.slice(ndarray::s![1..]).to_owned();
            (intercept, coeffs)
        } else {  (0.0, w) };
        Ok(LinearRegressionSolution { 
            coefficients, 
            intercept,
        })
    }
}

impl FitSupervised for LinearRegression {
    type Object = LinearRegressionSolution;
    fn fit(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Object> {
        self.clone().solve(SolveArgs { x, y })
    }
}
pub struct SolveArgs<'a, 'b> {
    pub x: MatrixView<'a>,
    pub y: VectorView<'b>,
}

#[derive(Debug, Clone)]
pub struct TrainingReport {}

impl fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Training Report (TODO)")
    }
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

    pub fn report(&self) -> Result<&TrainingReport> {
        Err(Error::invalid_state("Report not implemented"))
    }
}

impl Predictor for LinearRegressionSolution {
    fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        self.predict(x)
    }
}