pub use approx::assert_relative_eq;
pub use ndarray::{Array1, Array2};

pub type Matrix = Array2<f64>;
pub type Vector = Array1<f64>;

pub const STRICT_TOL: f64 = 1e-10;
pub const DEFAULT_TOL: f64 = 1e-6;
pub const LOOSE_TOL: f64 = 1e-3;

pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Assert two slices are approximately equal element-wise
pub fn assert_vec_approx_eq<A, E>(actual: A, expected: E, tol: f64)
where
    A: AsRef<[f64]>,
    E: AsRef<[f64]>,
{
    let actual = actual.as_ref();
    let expected = expected.as_ref();
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: actual={}, expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, tol),
            "Mismatch at index {}: actual={}, expected={}",
            i,
            a,
            e
        );
    }
}

/// Create a matrix from a flat slice with specified dimensions
pub fn make_matrix<S>(rows: usize, cols: usize, values: S) -> Matrix
where
    S: AsRef<[f64]>,
{
    let values = values.as_ref();
    assert_eq!(values.len(), rows * cols, "Values length must equal rows * cols");
    Matrix::from_shape_vec((rows, cols), values.to_vec()).unwrap()
}

/// Create a vector from a slice of values
pub fn make_vector<S>(values: S) -> Vector
where
    S: AsRef<[f64]>,
{
    Vector::from_vec(values.as_ref().to_vec())
}

/// Create a matrix filled with a constant value
pub fn make_filled_matrix(rows: usize, cols: usize, fill: f64) -> Matrix {
    Matrix::from_elem((rows, cols), fill)
}

/// Create a vector filled with a constant value
pub fn make_filled_vector(len: usize, fill: f64) -> Vector {
    Vector::from_elem(len, fill)
}

/// Generate linear data y = slope * x + intercept + noise
pub fn make_linear_data(n: usize, slope: f64, intercept: f64, noise: f64) -> (Matrix, Vector) {
    let mut x = Matrix::zeros((n, 1));
    let mut y = Vector::zeros(n);
    for i in 0..n {
        let xi = i as f64;
        x[[i, 0]] = xi;
        y[i] = slope * xi + intercept + noise * ((i % 3) as f64 - 1.0);
    }
    (x, y)
}

/// Generate perfect linear data without noise
pub fn make_perfect_linear_data(n: usize, slope: f64, intercept: f64) -> (Matrix, Vector) {
    make_linear_data(n, slope, intercept, 0.0)
}

/// Generate multivariate linear data with given coefficients
pub fn make_multifeature_linear_data<S>(
    n: usize,
    coefficients: S,
    intercept: f64,
) -> (Matrix, Vector)
where
    S: AsRef<[f64]>,
{
    let coefficients = coefficients.as_ref();
    let n_features = coefficients.len();
    let mut x = Matrix::zeros((n, n_features));
    let mut y = Vector::zeros(n);
    for i in 0..n {
        let mut yi = intercept;
        for j in 0..n_features {
            let xij = (i + j) as f64;
            x[[i, j]] = xij;
            yi += coefficients[j] * xij;
        }
        y[i] = yi;
    }
    (x, y)
}

/// Create a matrix with all elements set to the same value
pub fn make_constant_matrix(rows: usize, cols: usize, value: f64) -> Matrix {
    Matrix::from_elem((rows, cols), value)
}

/// Create a vector with all elements set to the same value
pub fn make_constant_vector(len: usize, value: f64) -> Vector {
    Vector::from_elem(len, value)
}

/// Check if all elements of two matrices are within tolerance
pub fn matrix_all_close(a: &Matrix, b: &Matrix, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if !approx_eq(a[[i, j]], b[[i, j]], tol) {
                return false;
            }
        }
    }
    true
}

/// Check if all elements of two vectors are within tolerance
pub fn vector_all_close(a: &Vector, b: &Vector, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (va, vb) in a.iter().zip(b.iter()) {
        if !approx_eq(*va, *vb, tol) {
            return false;
        }
    }
    true
}
