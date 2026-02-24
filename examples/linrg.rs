mod utils;

use naja::algorithms::linrg::{LinearRegression, Penalty};
use naja::core::traits::{Predictor, SupervisedEstimator};
use naja::metrics::{rmse, r2_score};
use utils::*;
use rand::thread_rng;
use std::time::Instant;

const N_SAMPLES: usize = 100_000;
const N_FEATURES: usize = 500;

fn main() {
    let mut rng = thread_rng();
    
    println!("=== Naja Linear Regression Benchmark ===");
    println!();
    
    let start = Instant::now();
    let (x, y, true_coef) = generate_massive_data(&mut rng, N_SAMPLES, N_FEATURES);
    let gen_time = start.elapsed();
    println!("[1] Generating data...");
    println!("    Shape: {} x {}", N_SAMPLES, N_FEATURES);
    println!("    Time: {:.3}ms", gen_time.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_ols = LinearRegression::new();
    let fitted_ols = model_ols.fit_supervised(x.view(), y.view()).unwrap();
    let time_ols = start.elapsed();
    println!("[2] Training OLS...");
    println!("    Time: {:.3}ms", time_ols.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_ridge = LinearRegression::new().penalty(Penalty::Ridge { alpha: 0.1 });
    let fitted_ridge = model_ridge.fit_supervised(x.view(), y.view()).unwrap();
    let time_ridge = start.elapsed();
    println!("[3] Training Ridge (alpha=0.1)...");
    println!("    Time: {:.3}ms", time_ridge.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_lasso = LinearRegression::new().penalty(Penalty::Lasso { alpha: 0.01 }).max_iter(1000);
    let fitted_lasso = model_lasso.fit_supervised(x.view(), y.view()).unwrap();
    let time_lasso = start.elapsed();
    println!("[4] Training Lasso (alpha=0.01)...");
    println!("    Time: {:.3}ms", time_lasso.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let pred_ols = fitted_ols.predict(x.view()).unwrap();
    let time_pred = start.elapsed();
    println!("[5] Predicting...");
    println!("    Time: {:.3}ms", time_pred.as_secs_f64() * 1e3);
    println!();
    
    let rmse_ols = rmse(y.view(), pred_ols.view()).unwrap();
    let r2_ols = r2_score(y.view(), pred_ols.view()).unwrap();
    
    let pred_ridge = fitted_ridge.predict(x.view()).unwrap();
    let rmse_ridge = rmse(y.view(), pred_ridge.view()).unwrap();
    let r2_ridge = r2_score(y.view(), pred_ridge.view()).unwrap();
    
    let pred_lasso = fitted_lasso.predict(x.view()).unwrap();
    let rmse_lasso = rmse(y.view(), pred_lasso.view()).unwrap();
    let r2_lasso = r2_score(y.view(), pred_lasso.view()).unwrap();
    
    println!("[6] Evaluation");
    println!("    {:<8} {:<14} {:<14} {:<14}", "Model", "RMSE", "R²", "Time");
    println!("    {}", separator(60));
    println!("    {:<8} {:<14.6} {:<14.6} {:.3}ms", "OLS", rmse_ols, r2_ols, time_ols.as_secs_f64() * 1e3);
    println!("    {:<8} {:<14.6} {:<14.6} {:.3}ms", "Ridge", rmse_ridge, r2_ridge, time_ridge.as_secs_f64() * 1e3);
    println!("    {:<8} {:<14.6} {:<14.6} {:.3}ms", "Lasso", rmse_lasso, r2_lasso, time_lasso.as_secs_f64() * 1e3);
    println!();
    
    println!("[7] Coefficient comparison (first 10)");
    println!("    {:<8} {:<12} {:<12} {:<12} {:<12}", "Index", "True", "OLS", "Ridge", "Lasso");
    println!("    {}", separator(60));
    for i in 0..10.min(N_FEATURES) {
        println!("    {:<8} {:<12.6} {:<12.6} {:<12.6} {:<12.6}", 
            i, true_coef[i], fitted_ols.coefficients()[i], 
            fitted_ridge.coefficients()[i], fitted_lasso.coefficients()[i]);
    }
    println!();
    
    let sparse_count = fitted_lasso.coefficients().iter().filter(|&&c| c.abs() < 1e-4).count();
    println!("[8] Lasso sparsity: {}/{} coefficients < 1e-4", sparse_count, N_FEATURES);
}
