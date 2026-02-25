mod utils;

use naja::algorithms::logrg::{LogisticRegression, Penalty, Solver};
use naja::core::traits::{Predictor, SupervisedEstimator};
use naja::metrics::{accuracy, precision, recall, f1_score};
use utils::*;
use rand::thread_rng;
use std::time::Instant;

const N_SAMPLES: usize = 50_000;
const N_FEATURES: usize = 100;

fn main() {
    let mut rng = thread_rng();
    
    println!("=== Naja Logistic Regression Benchmark ===");
    println!();
    
    let start = Instant::now();
    let (x, y, true_coef) = generate_logistic_data(&mut rng, N_SAMPLES, N_FEATURES);
    let gen_time = start.elapsed();
    println!("[1] Generating binary classification data...");
    println!("    Shape: {} x {}", N_SAMPLES, N_FEATURES);
    println!("    Positive class ratio: {:.2}%", 
        y.iter().filter(|&&v| v == 1.0).count() as f64 / N_SAMPLES as f64 * 100.0);
    println!("    Time: {:.3}ms", gen_time.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_none = LogisticRegression::new().learning_rate(0.1).max_iter(1000);
    let fitted_none = model_none.fit_supervised(x.view(), y.view()).unwrap();
    let time_none = start.elapsed();
    println!("[2] Training without penalty...");
    println!("    Time: {:.3}ms", time_none.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_ridge = LogisticRegression::new()
        .penalty(Penalty::Ridge { alpha: 0.1 })
        .learning_rate(0.1);
    let fitted_ridge = model_ridge.fit_supervised(x.view(), y.view()).unwrap();
    let time_ridge = start.elapsed();
    println!("[3] Training Ridge (alpha=0.1)...");
    println!("    Time: {:.3}ms", time_ridge.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let model_lasso = LogisticRegression::new()
        .penalty(Penalty::Lasso { alpha: 0.01 })
        .max_iter(1000)
        .solver(Solver::CoordinateDescent);
    let fitted_lasso = model_lasso.fit_supervised(x.view(), y.view()).unwrap();
    let time_lasso = start.elapsed();
    println!("[4] Training Lasso (alpha=0.01)...");
    println!("    Time: {:.3}ms", time_lasso.as_secs_f64() * 1e3);
    println!();
    
    let start = Instant::now();
    let pred_none = fitted_none.predict(x.view()).unwrap();
    let proba_none = fitted_none.predict_proba(x.view()).unwrap();
    let time_pred = start.elapsed();
    println!("[5] Predicting...");
    println!("    Time: {:.3}ms", time_pred.as_secs_f64() * 1e3);
    println!();
    
    let pred_ridge = fitted_ridge.predict(x.view()).unwrap();
    let pred_lasso = fitted_lasso.predict(x.view()).unwrap();
    
    let acc_none = accuracy(y.view(), pred_none.view()).unwrap();
    let prec_none = precision(y.view(), pred_none.view(), 1.0).unwrap();
    let rec_none = recall(y.view(), pred_none.view(), 1.0).unwrap();
    let f1_none = f1_score(y.view(), pred_none.view(), 1.0).unwrap();
    
    let acc_ridge = accuracy(y.view(), pred_ridge.view()).unwrap();
    let prec_ridge = precision(y.view(), pred_ridge.view(), 1.0).unwrap();
    let rec_ridge = recall(y.view(), pred_ridge.view(), 1.0).unwrap();
    let f1_ridge = f1_score(y.view(), pred_ridge.view(), 1.0).unwrap();
    
    let acc_lasso = accuracy(y.view(), pred_lasso.view()).unwrap();
    let prec_lasso = precision(y.view(), pred_lasso.view(), 1.0).unwrap();
    let rec_lasso = recall(y.view(), pred_lasso.view(), 1.0).unwrap();
    let f1_lasso = f1_score(y.view(), pred_lasso.view(), 1.0).unwrap();
    
    println!("[6] Evaluation");
    println!("    {:<8} {:<10} {:<10} {:<10} {:<10} {:<12}", 
        "Model", "Accuracy", "Precision", "Recall", "F1", "Time");
    println!("    {}", table_separator(&[8, 10, 10, 10, 10, 12]));
    println!("    {:<8} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:.3}ms", 
        "None", acc_none, prec_none, rec_none, f1_none, time_none.as_secs_f64() * 1e3);
    println!("    {:<8} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:.3}ms", 
        "Ridge", acc_ridge, prec_ridge, rec_ridge, f1_ridge, time_ridge.as_secs_f64() * 1e3);
    println!("    {:<8} {:<10.4} {:<10.4} {:<10.4} {:<10.4} {:.3}ms", 
        "Lasso", acc_lasso, prec_lasso, rec_lasso, f1_lasso, time_lasso.as_secs_f64() * 1e3);
    println!();
    
    println!("[7] Coefficient comparison (first 10)");
    println!("    {:<8} {:<12} {:<12} {:<12} {:<12}", 
        "Index", "True", "None", "Ridge", "Lasso");
    println!("    {}", table_separator(&[8, 12, 12, 12, 12]));
    for i in 0..10.min(N_FEATURES) {
        println!("    {:<8} {:<12.6} {:<12.6} {:<12.6} {:<12.6}", 
            i, true_coef[i], fitted_none.coefficients()[i], 
            fitted_ridge.coefficients()[i], fitted_lasso.coefficients()[i]);
    }
    println!();
    
    let sparse_count = fitted_lasso.coefficients().iter().filter(|&&c| c.abs() < 1e-4).count();
    println!("[8] Lasso sparsity: {}/{} coefficients < 1e-4", sparse_count, N_FEATURES);
    println!();
    
    println!("[9] Probability sample (first 5)");
    println!("    {:<8} {:<12} {:<12}", "Index", "Probability", "Predicted");
    println!("    {}", table_separator(&[8, 12, 12]));
    for i in 0..5 {
        println!("    {:<8} {:<12.6} {:<12}", i, proba_none[i], pred_none[i] as i32);
    }
}
