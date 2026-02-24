use ndarray::{Array1, Array2};
use rand::Rng;

pub fn generate_massive_data(
    rng: &mut impl Rng,
    n_samples: usize,
    n_features: usize,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array1::zeros(n_samples);
    let mut coef = Array1::zeros(n_features);
    
    for j in 0..n_features {
        coef[j] = rng.gen_range(-1.0..1.0);
    }
    
    for i in 0..n_samples {
        let mut yi = 0.0;
        for j in 0..n_features {
            x[[i, j]] = rng.gen_range(-1.0..1.0);
            yi += x[[i, j]] * coef[j];
        }
        y[i] = yi + rng.gen_range(-0.1..0.1);
    }
    (x, y, coef)
}

pub fn separator(width: usize) -> String {
    "═".repeat(width)
}

#[allow(dead_code)]
fn main() {}