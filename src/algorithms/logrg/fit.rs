use ndarray::Array1;
use std::collections::VecDeque;
use rand::{SeedableRng};
use rand::prelude::SliceRandom;
use rand_chacha::ChaCha8Rng;
use crate::core::Result;
use crate::core::compute::types::{MatrixView, Vector, VectorView};
use crate::core::compute::ops;
use crate::core::traits::Fitted;
use super::model::{LogisticRegression, LogisticRegressionConfig, Penalty, Solver};

pub fn fit(cfg: &LogisticRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LogisticRegression<Fitted>> {
    ops::ensure_nonempty_mat(x)?;
    ops::ensure_nonempty_vec(y)?;
    ops::ensure_len(y, x.column(0), "y", "x(rows)")?;
    validate_solver_penalty(&cfg.solver, &cfg.penalty)?;
    let x_design = if cfg.intercept { ops::add_intercept(x)? } else { x.to_owned() };
    
    match &cfg.solver {
        Solver::GradientDescent { learning_rate, grad_tol } => {
            solve_gradient_descent(cfg, x_design.view(), y, *learning_rate, *grad_tol)
        }
        Solver::Irls => solve_irls(cfg, x_design.view(), y),
        Solver::Sgd { batch_size, learning_rate } => {
            solve_sgd(cfg, x_design.view(), y, *batch_size, *learning_rate)
        }
        Solver::Lbfgs { memory_size } => {
            solve_lbfgs(cfg, x_design.view(), y, *memory_size)
        }
        Solver::CoordinateDescent => solve_coordinate_descent(cfg, x_design.view(), y),
    }
}

fn validate_solver_penalty(solver: &Solver, penalty: &Penalty) -> Result<()> {
    match (solver, penalty) {
        (Solver::CoordinateDescent, Penalty::None) => 
            Err(crate::core::Error::invalid_param("solver", "CoordinateDescent requires regularization (Ridge or Lasso)")),
        _ => Ok(())
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
    else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn compute_gradient(x: MatrixView<'_>, y: VectorView<'_>, w: &Vector, penalty: Penalty) -> Result<Vector> {
    let n = x.nrows() as f64;
    
    let z = ops::gemv(x, w.view())?;
    let p = z.mapv(sigmoid);
    let y_owned = y.to_owned();
    let diff = p - y_owned;
    let mut grad = ops::gemv(x.t().view(), diff.view())?;
    grad.mapv_inplace(|g| g / n);
    
    if let Penalty::Ridge { alpha } = penalty {
        for j in 1..w.len() {
            grad[j] += alpha * w[j];
        }
    }
    
    Ok(grad)
}

fn compute_loss(x: MatrixView<'_>, y: VectorView<'_>, w: &Vector, penalty: Penalty) -> Result<f64> {
    let n = x.nrows() as f64;
    let z = ops::gemv(x, w.view())?;
    let mut loss = 0.0;
    for i in 0..x.nrows() {
        let p = sigmoid(z[i]);
        let eps = 1e-15;
        let p_clipped = p.clamp(eps, 1.0 - eps);
        loss -= y[i] * p_clipped.ln() + (1.0 - y[i]) * (1.0 - p_clipped).ln();
    }
    loss /= n;
    
    if let Penalty::Ridge { alpha } = penalty {
        let reg: f64 = w.iter().skip(1).map(|wi| 0.5 * alpha * wi * wi).sum();
        loss += reg;
    }
    
    Ok(loss)
}

fn solve_gradient_descent(
    cfg: &LogisticRegressionConfig, 
    x: MatrixView<'_>, 
    y: VectorView<'_>,
    learning_rate: f64,
    grad_tol: f64,
) -> Result<LogisticRegression<Fitted>> {
    let n_features = x.ncols();
    let mut w = Array1::<f64>::zeros(n_features);
    let mut prev_loss = f64::INFINITY;
    
    for _iter in 0..cfg.max_iter {
        let grad = compute_gradient(x, y, &w, cfg.penalty)?;
        
        let grad_norm = ops::l2(grad.view())?;
        if grad_norm < grad_tol {
            break;
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

fn solve_irls(cfg: &LogisticRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LogisticRegression<Fitted>> {
    let n_features = x.ncols();
    let mut w = Array1::<f64>::zeros(n_features);
    
    for _iter in 0..cfg.max_iter {
        let z = ops::gemv(x, w.view())?;
        let p = z.mapv(sigmoid);
        let p_clipped = p.mapv(|pi| pi.clamp(1e-10, 1.0 - 1e-10));
        
        let w_diag = p_clipped.mapv(|pi| pi * (1.0 - pi));
        
        let working_response = compute_working_response(x, y, &w, &p_clipped, &w_diag)?;
        let mut xtwx = ops::xtwx(x, w_diag.view())?;
        
        if let Penalty::Ridge { alpha } = cfg.penalty {
            for j in 1..n_features {
                xtwx[[j, j]] += alpha;
            }
        }
        
        let xtwz = ops::xtwz(x, w_diag.view(), working_response.view())?;
        
        let w_new = ops::solve_cholesky(xtwx.view(), xtwz.view())
            .or_else(|_| ops::solve_svd(xtwx.view(), xtwz.view()))?;
        
        let diff: f64 = w_new.iter().zip(w.iter()).map(|(a, b)| (a - b).abs()).sum();
        if diff < cfg.tol {
            w = w_new;
            break;
        }
        w = w_new;
    }
    
    package_solution(cfg, w)
}

fn compute_working_response(
    x: MatrixView<'_>,
    y: VectorView<'_>,
    w: &Vector,
    p: &Vector,
    w_diag: &Vector,
) -> Result<Vector> {
    let n = x.nrows();
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let z_i: f64 = x.row(i).iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum();
        z[i] = z_i + (y[i] - p[i]) / (w_diag[i] + 1e-10);
    }
    Ok(z)
}

fn solve_sgd(
    cfg: &LogisticRegressionConfig, 
    x: MatrixView<'_>, 
    y: VectorView<'_>,
    batch_size: usize,
    learning_rate: f64,
) -> Result<LogisticRegression<Fitted>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();
    let mut w = Array1::<f64>::zeros(n_features);
    let actual_batch = batch_size.max(1).min(n_samples);
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    
    let mut prev_loss = f64::INFINITY;
    let mut no_improve_count = 0;
    const PATIENCE: usize = 3;
    
    for epoch in 0..cfg.max_iter {
        indices.shuffle(&mut rng);
        
        for chunk in indices.chunks(actual_batch) {
            let mut grad = Array1::<f64>::zeros(n_features);
            let batch_n = chunk.len() as f64;
            
            for &i in chunk {
                let z: f64 = x.row(i).iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum();
                let p = sigmoid(z);
                let diff = p - y[i];
                for j in 0..n_features {
                    grad[j] += diff * x[[i, j]];
                }
            }
            
            grad.mapv_inplace(|g| g / batch_n);
            
            if let Penalty::Ridge { alpha } = cfg.penalty {
                for j in 1..n_features {
                    grad[j] += alpha * w[j];
                }
            }
            
            for j in 0..n_features {
                w[j] -= learning_rate * grad[j];
            }
        }
        
        if epoch % 10 == 0 || epoch == cfg.max_iter - 1 {
            let loss = compute_loss(x, y, &w, cfg.penalty)?;
            if (prev_loss - loss).abs() < cfg.tol {
                no_improve_count += 1;
                if no_improve_count >= PATIENCE {
                    break;
                }
            } else {
                no_improve_count = 0;
            }
            prev_loss = loss;
        }
    }
    
    package_solution(cfg, w)
}

fn solve_lbfgs(
    cfg: &LogisticRegressionConfig, 
    x: MatrixView<'_>, 
    y: VectorView<'_>,
    memory_size: usize,
) -> Result<LogisticRegression<Fitted>> {
    let n_features = x.ncols();
    let mut w = Array1::<f64>::zeros(n_features);
    
    let memory_size = memory_size.max(1);
    let mut s_queue: VecDeque<Array1<f64>> = VecDeque::with_capacity(memory_size);
    let mut y_queue: VecDeque<Array1<f64>> = VecDeque::with_capacity(memory_size);
    let mut rho_queue: VecDeque<f64> = VecDeque::with_capacity(memory_size);
    
    let mut prev_grad = compute_gradient(x, y, &w, cfg.penalty)?;
    let mut prev_loss = compute_loss(x, y, &w, cfg.penalty)?;
    
    for _iter in 0..cfg.max_iter {
        let grad = prev_grad.clone();
        
        let direction = lbfgs_two_loop(&grad, &s_queue, &y_queue, &rho_queue);
        
        let step = line_search(x, y, &w, &direction, &grad, &prev_loss, cfg.penalty)?;
        
        let s = direction.mapv(|d| d * step);
        let w_new = &w + &s;
        
        let grad_new = compute_gradient(x, y, &w_new, cfg.penalty)?;
        let y_diff = &grad_new - &grad;
        
        let sy = ops::dot(s.view(), y_diff.view())?;
        if sy > 1e-10 {
            if s_queue.len() >= memory_size {
                s_queue.pop_front();
                y_queue.pop_front();
                rho_queue.pop_front();
            }
            s_queue.push_back(s);
            y_queue.push_back(y_diff.clone());
            rho_queue.push_back(1.0 / sy);
        }
        
        if ops::l2(grad_new.view())? < cfg.tol {
            w = w_new;
            break;
        }
        
        w = w_new;
        prev_grad = grad_new;
        prev_loss = compute_loss(x, y, &w, cfg.penalty)?;
    }
    
    package_solution(cfg, w)
}

fn lbfgs_two_loop(
    grad: &Vector,
    s_queue: &VecDeque<Array1<f64>>,
    y_queue: &VecDeque<Array1<f64>>,
    rho_queue: &VecDeque<f64>,
) -> Vector {
    let n = grad.len();
    let m = s_queue.len();
    
    if m == 0 {
        return grad.mapv(|g| -g);
    }
    
    let mut q = grad.clone();
    let mut alpha_list = Vec::with_capacity(m);
    
    for i in (0..m).rev() {
        let alpha_i = rho_queue[i] * ops::dot(s_queue[i].view(), q.view()).unwrap_or(0.0);
        alpha_list.push(alpha_i);
        for j in 0..n {
            q[j] -= alpha_i * y_queue[i][j];
        }
    }
    
    let s_last = s_queue.front().unwrap();
    let y_last = y_queue.front().unwrap();
    let sy = ops::dot(s_last.view(), y_last.view()).unwrap_or(1.0);
    let yy = ops::dot(y_last.view(), y_last.view()).unwrap_or(1.0);
    let gamma = sy / yy.max(1e-10);
    
    let mut r = q.mapv(|qi| -gamma * qi);
    
    let mut alpha_iter = alpha_list.into_iter().rev();
    for i in 0..m {
        let beta = rho_queue[m - 1 - i] * ops::dot(y_queue[m - 1 - i].view(), r.view()).unwrap_or(0.0);
        let alpha_i = alpha_iter.next().unwrap_or(0.0);
        for j in 0..n {
            r[j] += s_queue[m - 1 - i][j] * (alpha_i - beta);
        }
    }
    
    r
}

fn line_search(
    x: MatrixView<'_>,
    y: VectorView<'_>,
    w: &Vector,
    direction: &Vector,
    grad: &Vector,
    prev_loss: &f64,
    penalty: Penalty,
) -> Result<f64> {
    let mut step = 1.0;
    let c1 = 1e-4;
    let dginit = ops::dot(direction.view(), grad.view()).unwrap_or(0.0);
    
    for _ in 0..20 {
        let w_new = w + &direction.mapv(|d| d * step);
        let loss = compute_loss(x, y, &w_new, penalty)?;
        
        if loss <= *prev_loss + c1 * step * dginit {
            return Ok(step);
        }
        step *= 0.5;
    }
    
    Ok(step)
}

fn solve_coordinate_descent(cfg: &LogisticRegressionConfig, x: MatrixView<'_>, y: VectorView<'_>) -> Result<LogisticRegression<Fitted>> {
    let n_features = x.ncols();
    let n_samples = x.nrows();
    let mut w = Array1::<f64>::zeros(n_features);
    let alpha = match cfg.penalty {
        Penalty::Lasso { alpha } => alpha,
        Penalty::Ridge { alpha } => alpha,
        Penalty::None => 0.0,
    };
    
    for _iter in 0..cfg.max_iter {
        let mut max_change: f64 = 0.0;
        for j in 0..n_features {
            let is_intercept = cfg.intercept && j == 0;
            let mut rho: f64 = 0.0;
            let mut norm_sq: f64 = 0.0;
            for i in 0..n_samples {
                let z_ij = x[[i, j]];
                let z: f64 = x.row(i).iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum();
                let p = sigmoid(z);
                rho += z_ij * (y[i] - p + w[j] * z_ij);
                norm_sq += z_ij * z_ij;
            }
            if norm_sq == 0.0 { continue; }
            let old_w_j = w[j];
            let new_w_j;
            if is_intercept {
                new_w_j = rho / norm_sq;
            } else {
                match cfg.penalty {
                    Penalty::Lasso { .. } => {
                        let soft_thresholded = ops::soft_threshold(rho, alpha * n_samples as f64);
                        new_w_j = soft_thresholded / norm_sq;
                    }
                    Penalty::Ridge { .. } => {
                        new_w_j = rho / (norm_sq + alpha * n_samples as f64);
                    }
                    Penalty::None => {
                        new_w_j = rho / norm_sq;
                    }
                }
            }
            let diff = new_w_j - old_w_j;
            if diff.abs() > max_change {
                max_change = diff.abs();
            }
            w[j] = new_w_j;
        }
        if max_change < cfg.tol { break; }
    }
    package_solution(cfg, w)
}

fn package_solution(cfg: &LogisticRegressionConfig, w: Vector) -> Result<LogisticRegression<Fitted>> {
    let (intercept_value, coefficients) = if cfg.intercept {
        (w[0], w.slice(ndarray::s![1..]).to_owned())
    } else { (0.0, w) };
    Ok(LogisticRegression::new_fitted(cfg.clone(), coefficients, intercept_value))
}
