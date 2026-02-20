use naja::algorithms::linrg::LinearRegression;
use naja::core::regularization::Penalty;
use naja::core::traits::FitSupervised;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Naja Linear Regression Large Scale Demo");

    let n_samples = 100_000;
    let n_features = 50;
    let noise_std = 1.0;
    println!("Dataset size: {} samples, {} features", n_samples, n_features);

    let mut rng = rand::thread_rng();
    let true_weights: Array1<f64> = Array1::from_iter((0..n_features).map(|_| rng.gen_range(-5.0..5.0)));
    let true_intercept = rng.gen_range(-10.0..10.0);

    let mut x_data = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples { for j in 0..n_features { x_data[[i, j]] = rng.gen_range(-10.0..10.0); } }

    let mut y_data = x_data.dot(&true_weights) + true_intercept;
    for i in 0..n_samples { y_data[i] += rng.gen_range(-noise_std..noise_std); }

    let n_train = (n_samples as f64 * 0.8) as usize;
    let x_train = x_data.slice(ndarray::s![0..n_train, ..]);
    let y_train = y_data.slice(ndarray::s![0..n_train]);
    let x_test = x_data.slice(ndarray::s![n_train.., ..]);
    let y_test = y_data.slice(ndarray::s![n_train..]);
    println!("Data generation complete. Training on {} samples...", n_train);

    let start_time = Instant::now();
    let model = LinearRegression::new()
        .intercept(true)
        .penalty(Penalty::None);
    let solution = model.fit(x_train, y_train)?;
    let duration = start_time.elapsed();
    println!("Training completed in {:.2?}", duration);

    let y_pred = solution.predict(x_test)?;
    let residuals = &y_test - &y_pred;
    let mse = residuals.mapv(|x| x.powi(2)).mean().unwrap_or(0.0);
    let mae = residuals.mapv(|x| x.abs()).mean().unwrap_or(0.0);
    let y_test_mean = y_test.mean().unwrap_or(0.0);
    let ss_tot = y_test.mapv(|x| (x - y_test_mean).powi(2)).sum();
    let ss_res = residuals.mapv(|x| x.powi(2)).sum();
    let r2 = 1.0 - (ss_res / ss_tot);
    println!("\nModel Performance (Test Set):");
    println!("MSE: {:.6}", mse);
    println!("MAE: {:.6}", mae);
    println!("R² : {:.6}", r2);
    
    let weight_diff = &solution.coefficients - &true_weights;
    let max_weight_error = weight_diff.mapv(|x| x.abs()).iter().cloned().fold(0.0, f64::max);
    let intercept_error = (solution.intercept - true_intercept).abs();
    println!("\nParameter Recovery:");
    println!("Max Weight Error: {:.6}", max_weight_error);
    println!("Intercept Error : {:.6}", intercept_error);
    
    Ok(())
}
