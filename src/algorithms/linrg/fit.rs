use ndarray::Array1;
use crate::core::Result;
use crate::core::compute::types::{MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::traits::Fitted;
use super::model::{LinearRegression, LinearRegressionConfig, Penalty, Solver};

pub fn fit(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegression<Fitted>> {
    ops::ensure_nonempty_mat(x)?;
    ops::ensure_nonempty_vec(y)?;
    ops::ensure_len(y, x.column(0), "y", "x(rows)")?;
    validate_solver_penalty(&cfg.solver, &cfg.penalty)?;
    let x_design = if cfg.intercept { ops::add_intercept(x)? } else { x.to_owned() };
    
    match &cfg.solver {
        Solver::ClosedForm => solve_closed_form(cfg, x_design.view(), y),
        Solver::GradientDescent { learning_rate, grad_tol } => {
            solve_gradient_descent(cfg, x_design.view(), y, *learning_rate, *grad_tol)
        }
        Solver::CoordinateDescent => solve_coordinate_descent(cfg, x_design.view(), y),
    }
}

fn validate_solver_penalty(solver: &Solver, penalty: &Penalty) -> Result<()> {
    match (solver, penalty) {
        (Solver::ClosedForm, Penalty::Lasso { .. }) => 
            Err(crate::core::Error::invalid_param("solver", "ClosedForm does not support Lasso, use CoordinateDescent")),
        _ => Ok(())
    }
}

fn apply_ridge(xtx: &mut ndarray::Array2<f64>, alpha: f64, intercept: bool) {
    let start_idx = if intercept { 1 } else { 0 };
    for i in start_idx..xtx.nrows() { xtx[[i, i]] += alpha; }
}

fn solve_closed_form(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegression<Fitted>> {
    let mut xtx = ops::xtx(x)?;
    if let Penalty::Ridge { alpha } = cfg.penalty { apply_ridge(&mut xtx, alpha, cfg.intercept); }
    let xty = ops::xty(x, y)?;
    let w = ops::solve_cholesky(xtx.view(), xty.view()).or_else(|_| ops::solve_svd(xtx.view(), xty.view()))?;
    package_solution(cfg, w)
}

fn compute_gradient(x: MatrixView<'_>, y: VectorView<'_>, w: &Vector) -> Result<Vector> {
    let n = x.nrows() as f64;
    let pred = ops::gemv(x, w.view())?;
    let y_owned = y.to_owned();
    let diff = pred - y_owned;
    let grad = ops::gemv(x.t().view(), diff.view())?;
    Ok(grad.mapv(|g| g / n))
}

fn solve_gradient_descent(
    cfg: &LinearRegressionConfig, 
    x: MatrixView<'_>, 
    y: VectorView<'_>,
    learning_rate: f64,
    grad_tol: f64,
) -> Result<LinearRegression<Fitted>> {
    let n_features = x.ncols();
    let mut w = Array1::<f64>::zeros(n_features);
    let mut prev_loss = f64::INFINITY;
    
    for _ in 0..cfg.max_iter {
        let mut grad = compute_gradient(x, y, &w)?;
        
        let grad_norm = ops::l2(grad.view())?;
        if grad_norm < grad_tol {
            break;
        }
        
        if let Penalty::Ridge { alpha } = cfg.penalty {
            let start_idx = if cfg.intercept { 1 } else { 0 };
            for j in start_idx..n_features {
                grad[j] += alpha * w[j];
            }
        }
        
        for j in 0..n_features {
            w[j] -= learning_rate * grad[j];
        }
        
        let loss = compute_loss(x, y, &w, cfg.penalty)?;
        if (prev_loss - loss).abs() < cfg.tol {
            break;
        }
        prev_loss = loss;
    }
    
    package_solution(cfg, w)
}

fn compute_loss(x: MatrixView<'_>, y: VectorView<'_>, w: &Vector, penalty: Penalty) -> Result<f64> {
    let n = x.nrows() as f64;
    let pred = ops::gemv(x, w.view())?;
    let y_owned = y.to_owned();
    let diff = pred - y_owned;
    let loss: f64 = diff.mapv(|e| e * e).sum() / n;
    
    if let Penalty::Ridge { alpha } = penalty {
        let start_idx = 1;
        let reg: f64 = w.iter().skip(start_idx).map(|wi| alpha * wi * wi).sum();
        return Ok(loss + reg);
    }
    
    Ok(loss)
}

fn solve_coordinate_descent(cfg: &LinearRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LinearRegression<Fitted>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();
    let mut w = Array1::<f64>::zeros(n_features);
    let mut norm_sq = Array1::<f64>::zeros(n_features);
    for j in 0..n_features { norm_sq[j] = ops::dot(x.column(j), x.column(j))?; }
    let mut r = y.to_owned();
    
    let alpha = match cfg.penalty {
        Penalty::Lasso { alpha } => alpha,
        Penalty::Ridge { alpha } => alpha,
        Penalty::None => 0.0,
    };
    
    for _iter in 0..cfg.max_iter {
        let mut max_change: f64 = 0.0;
        for j in 0..n_features {
            let is_intercept = cfg.intercept && j == 0;
            if norm_sq[j] == 0.0 { continue; }
            let dot_xr = ops::dot(x.column(j), r.view())?;
            let rho = dot_xr + w[j] * norm_sq[j];
            let old_w_j = w[j];
            let new_w_j;
            
            if is_intercept {
                new_w_j = rho / norm_sq[j];
            } else {
                match cfg.penalty {
                    Penalty::Lasso { .. } => {
                        let soft_thresholded = ops::soft_threshold(rho, alpha * n_samples as f64);
                        new_w_j = soft_thresholded / norm_sq[j];
                    }
                    Penalty::Ridge { .. } => {
                        new_w_j = rho / (norm_sq[j] + alpha * n_samples as f64);
                    }
                    Penalty::None => {
                        new_w_j = rho / norm_sq[j];
                    }
                }
            }
            
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

fn package_solution(cfg: &LinearRegressionConfig, w: Vector) -> Result<LinearRegression<Fitted>> {
    let (intercept_value, coefficients) = if cfg.intercept {
        (w[0], w.slice(ndarray::s![1..]).to_owned())
    } else { (0.0, w) };
    Ok(LinearRegression::new_fitted(cfg.clone(), coefficients, intercept_value))
}
