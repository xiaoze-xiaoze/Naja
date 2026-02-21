use ndarray::Array1;
use crate::core::{Result, Error};
use crate::core::compute::types::{MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::regularization::Penalty;
use super::model::LinearRegressionConfig;
use super::predict::LinearRegressionSolution;

pub fn fit(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegressionSolution> {
    ops::ensure_nonempty_mat(x)?;
    ops::ensure_nonempty_vec(y)?;
    ops::ensure_len(y, x.column(0), "y", "x(rows)")?;
    let x_design = if cfg.intercept { ops::add_intercept(x)? } else { x.to_owned() };
    match cfg.penalty {
        Penalty::None | Penalty::Ridge { .. } => solve_closed_form(cfg, x_design.view(), y),
        Penalty::Lasso { .. } => solve_lasso(cfg, x_design.view(), y),
    }
}

fn solve_closed_form(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegressionSolution> {
    let mut xtx = ops::xtx(x)?;
    cfg.penalty.apply_l2(&mut xtx, cfg.intercept);
    let xty = ops::xty(x, y)?;
    let w = ops::solve_cholesky(xtx.view(), xty.view()).map_err(|e| Error::lin_alg(format!("Failed to solve normal equation: {}", e)))?;
    package_solution(cfg, w)
}

fn solve_lasso(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegressionSolution> {
    let n_features = x.ncols();
    let mut w = Array1::<f64>::zeros(n_features);
    let mut norm_sq = Array1::<f64>::zeros(n_features);
    for j in 0..n_features { norm_sq[j] = ops::dot(x.column(j), x.column(j))?; }
    let mut r = y.to_owned();
    for _iter in 0..cfg.max_iter {
        let mut max_change: f64 = 0.0;
        for j in 0..n_features {
            let is_intercept = cfg.intercept && j == 0;
            if norm_sq[j] == 0.0 { continue; }
            let dot_xr = ops::dot(x.column(j), r.view())?;
            let rho = dot_xr + w[j] * norm_sq[j];
            let old_w_j = w[j];
            let new_w_j;
            if is_intercept { new_w_j = rho / norm_sq[j]; } 
            else { new_w_j = cfg.penalty.apply_l1(rho) / norm_sq[j]; }
            if (new_w_j - old_w_j).abs() > 1e-15 {
                let diff = new_w_j - old_w_j;
                ops::add_scaled_mut(&mut r, x.column(j), -diff)?;
                w[j] = new_w_j;
                max_change = max_change.max(diff.abs());
            }
        }
        if max_change < cfg.tol { break; }
    }
    
    package_solution(cfg, w)
}

fn package_solution(cfg: &LinearRegressionConfig, w: Vector) -> Result<LinearRegressionSolution> {
    let (intercept, coefficients) = if cfg.intercept {
        let intercept = w[0];
        let coeffs = w.slice(ndarray::s![1..]).to_owned();
        (intercept, coeffs)
    } else {  (0.0, w) };
    Ok(LinearRegressionSolution { coefficients, intercept })
}
