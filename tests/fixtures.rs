use ndarray::{Array1, Array2};

pub fn simple_linear_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Array1::from_vec(vec![5.0, 7.0, 9.0, 11.0, 13.0]);
    (x, y)
}

pub fn simple_linear_with_intercept() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    (x, y)
}

pub fn multivariate_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((4, 2), vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
    ]).unwrap();
    let y = Array1::from_vec(vec![2.0, 3.0, 5.0, 7.0]);
    (x, y)
}

pub fn multicollinear_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((5, 3), vec![
        1.0, 1.0, 2.0,
        2.0, 2.0, 4.0,
        3.0, 3.0, 6.0,
        4.0, 4.0, 8.0,
        5.0, 5.0, 10.0,
    ]).unwrap();
    let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
    (x, y)
}

pub fn sparse_true_coeffs() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((10, 5), vec![
        0.1, 0.0, 0.3, 0.0, 0.5,
        0.2, 0.0, 0.6, 0.0, 0.1,
        0.3, 0.0, 0.1, 0.0, 0.8,
        0.4, 0.0, 0.2, 0.0, 0.3,
        0.5, 0.0, 0.5, 0.0, 0.2,
        0.6, 0.0, 0.4, 0.0, 0.6,
        0.7, 0.0, 0.3, 0.0, 0.4,
        0.8, 0.0, 0.7, 0.0, 0.1,
        0.9, 0.0, 0.2, 0.0, 0.9,
        1.0, 0.0, 0.8, 0.0, 0.2,
    ]).unwrap();
    let y = Array1::from_vec(vec![0.9, 0.6, 1.3, 0.7, 1.1, 1.1, 1.2, 1.2, 1.7, 1.5]);
    (x, y)
}

pub fn noisy_linear_data(seed: u64) -> (Array2<f64>, Array1<f64>) {
    let n = 50;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);
    
    let mut rng_state = seed;
    let mut rand_val = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };
    
    for i in 0..n {
        let x_val = i as f64 * 0.1;
        x_data.push(x_val);
        let noise = (rand_val() - 0.5) * 0.5;
        y_data.push(2.0 * x_val + 1.0 + noise);
    }
    
    let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

pub fn single_sample_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let y = Array1::from_vec(vec![5.0]);
    (x, y)
}

pub fn two_samples_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0]);
    (x, y)
}

pub fn empty_x() -> Array2<f64> {
    Array2::from_shape_vec((0, 2), vec![] as Vec<f64>).unwrap()
}

pub fn empty_y() -> Array1<f64> {
    Array1::from_vec(vec![])
}

pub fn mismatched_len_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
    let y = Array1::from_vec(vec![1.0, 2.0]);
    (x, y)
}
