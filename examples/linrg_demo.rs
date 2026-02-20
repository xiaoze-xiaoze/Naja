use naja::algorithms::linrg::LinearRegression;
use naja::core::regularization::Penalty;
use naja::core::traits::FitSupervised;
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Naja Linear Regression Demo:");

    let (n_samples, n_features, noise_std) = (1000, 5, 0.5);
    println!("Generating dataset: {} samples, {} features...", n_samples, n_features);

    let true_weights = Array1::from_vec(vec![2.0, -3.0, 0.5, 1.5, -1.0]);
    let true_intercept = 10.0;
    
    println!("True Weights: {}\nTrue Intercept: {}", true_weights, true_intercept);

    let mut rng = rand::thread_rng();
    let mut x_train = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples { for j in 0..n_features { x_train[[i, j]] = rng.gen_range(-5.0..5.0); } }

    let mut y_train = x_train.dot(&true_weights) + true_intercept;
    for i in 0..n_samples { y_train[i] += rng.gen_range(-noise_std..noise_std); }

    println!("\nTraining model...");
    let model = LinearRegression::new().intercept(true).penalty(Penalty::None);
    let solution = model.fit(x_train.view(), y_train.view())?;

    println!("\nModel Trained!");
    println!("Estimated Intercept: {:.4}", solution.intercept);
    println!("Estimated Weights: {}", solution.coefficients);

    let diff_weights = &solution.coefficients - &true_weights;
    let mae_weights = diff_weights.mapv(|v| v.abs()).sum() / n_features as f64;
    println!("MAE of Weights: {:.6}", mae_weights);

    // Predict on some new data
    let x_test = x_train.slice(ndarray::s![0..5, ..]);
    let y_pred = solution.predict(x_test)?;
    println!("\nPredictions on first 5 samples: {}", y_pred);
    
    Ok(())
}
