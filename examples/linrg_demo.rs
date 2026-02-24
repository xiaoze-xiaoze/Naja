use std::time::Instant;
use naja::algorithms::linrg::{LinearRegression, Penalty};
use naja::core::traits::{FitSupervised, Predictor};
use naja::metrics;

mod utils;
use utils::generate_linear_data;

fn main() {
    let n_samples = 500_000;
    let n_features = 200;
    let noise_level = 0.1;
    let seed = 42u64;

    println!("=== Naja Linear Regression Demo ===\n");

    println!("[1] Generating data...");
    let start = Instant::now();
    let (x, y, true_coeffs) = generate_linear_data(n_samples, n_features, noise_level, seed);
    println!("    Shape: {} × {}", n_samples, n_features);
    println!("    Time: {:?}\n", start.elapsed());

    println!("[2] Training OLS...");
    let start = Instant::now();
    let ols = LinearRegression::new()
        .intercept(false)
        .fit(x.view(), y.view())
        .expect("OLS fit failed");
    let ols_time = start.elapsed();
    println!("    Time: {:?}\n", ols_time);

    println!("[3] Training Ridge (alpha=0.1)...");
    let start = Instant::now();
    let ridge = LinearRegression::new()
        .intercept(false)
        .penalty(Penalty::Ridge { alpha: 0.1 })
        .fit(x.view(), y.view())
        .expect("Ridge fit failed");
    let ridge_time = start.elapsed();
    println!("    Time: {:?}\n", ridge_time);

    println!("[4] Training Lasso (alpha=0.01)...");
    let start = Instant::now();
    let lasso = LinearRegression::new()
        .intercept(false)
        .penalty(Penalty::Lasso { alpha: 0.01 })
        .max_iter(5000)
        .tol(1e-6)
        .fit(x.view(), y.view())
        .expect("Lasso fit failed");
    let lasso_time = start.elapsed();
    println!("    Time: {:?}\n", lasso_time);

    println!("[5] Predicting...");
    let start = Instant::now();
    let ols_pred = ols.predict(x.view()).unwrap();
    let ridge_pred = ridge.predict(x.view()).unwrap();
    let lasso_pred = lasso.predict(x.view()).unwrap();
    println!("    Time: {:?}\n", start.elapsed());

    println!("[6] Evaluation");
    println!("{}", "─".repeat(43));
    println!("{:<10} {:>10} {:>10} {:>10}", "Model", "RMSE", "R²", "Time");
    println!("{}", "─".repeat(43));
    let ols_m = metrics::RegressionMetrics::new(y.view(), ols_pred.view()).unwrap();
    println!("{:<10} {:>10.6} {:>10.6} {:>10?}", "OLS", ols_m.rmse, ols_m.r2, ols_time);
    let ridge_m = metrics::RegressionMetrics::new(y.view(), ridge_pred.view()).unwrap();
    println!("{:<10} {:>10.6} {:>10.6} {:>10?}", "Ridge", ridge_m.rmse, ridge_m.r2, ridge_time);
    let lasso_m = metrics::RegressionMetrics::new(y.view(), lasso_pred.view()).unwrap();
    println!("{:<10} {:>10.6} {:>10.6} {:>10?}", "Lasso", lasso_m.rmse, lasso_m.r2, lasso_time);
    println!("{}\n", "─".repeat(43));

    println!("[7] Coefficient comparison (first 5)");
    println!("{}", "─".repeat(58));
    println!("{:>6} {:>12} {:>12} {:>12} {:>12}", "Index", "True", "OLS", "Ridge", "Lasso");
    println!("{}", "─".repeat(58));
    for i in 0..5.min(n_features) {
        println!("{:>6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}", i, true_coeffs[i], ols.coefficients[i], ridge.coefficients[i], lasso.coefficients[i]);
    }
    println!("{}\n", "─".repeat(58));

    let zero_count = lasso.coefficients.iter().filter(|&&c| c.abs() < 1e-4).count();
    println!("[8] Lasso sparsity: {}/{} coefficients < 1e-4", zero_count, n_features);

    println!("\n=== Demo Complete ===");
}
