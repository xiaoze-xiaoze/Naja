use crate::common::*;
use naja::core::compute::ops;
use naja::core::Error;

mod basic {
    use super::*;

    #[test]
    fn test_dot() {
        let a = make_vector(&[1.0, 2.0, 3.0]);
        let b = make_vector(&[4.0, 5.0, 6.0]);
        let result = ops::dot(a.view(), b.view()).unwrap();
        approx_eq(result, 32.0, DEFAULT_TOL);
    }

    #[test]
    fn test_l2() {
        let v = make_vector(&[3.0, 4.0]);
        let result = ops::l2(v.view()).unwrap();
        approx_eq(result, 5.0, DEFAULT_TOL);
    }

    #[test]
    fn test_l2_sq() {
        let v = make_vector(&[3.0, 4.0]);
        let result = ops::l2_sq(v.view()).unwrap();
        approx_eq(result, 25.0, DEFAULT_TOL);
    }

    #[test]
    fn test_gemv() {
        let a = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = make_vector(&[1.0, 1.0, 1.0]);
        let result = ops::gemv(a.view(), x.view()).unwrap();
        assert_vec_approx_eq(&result.to_vec(), &[6.0, 15.0], DEFAULT_TOL);
    }

    #[test]
    fn test_matmul() {
        let a = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = ops::matmul(a.view(), b.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        approx_eq(result[[0, 0]], 22.0, DEFAULT_TOL);
        approx_eq(result[[0, 1]], 28.0, DEFAULT_TOL);
        approx_eq(result[[1, 0]], 49.0, DEFAULT_TOL);
        approx_eq(result[[1, 1]], 64.0, DEFAULT_TOL);
    }

    #[test]
    fn test_xtx() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = ops::xtx(x.view()).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        approx_eq(result[[0, 0]], 10.0, DEFAULT_TOL);
        approx_eq(result[[0, 1]], 14.0, DEFAULT_TOL);
        approx_eq(result[[1, 0]], 14.0, DEFAULT_TOL);
        approx_eq(result[[1, 1]], 20.0, DEFAULT_TOL);
    }

    #[test]
    fn test_xty() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = make_vector(&[5.0, 6.0]);
        let result = ops::xty(x.view(), y.view()).unwrap();
        assert_vec_approx_eq(&result.to_vec(), &[23.0, 34.0], DEFAULT_TOL);
    }

    #[test]
    fn test_col_mean() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = ops::col_mean(x.view()).unwrap();
        assert_vec_approx_eq(&result.to_vec(), &[2.0, 3.0], DEFAULT_TOL);
    }

    #[test]
    fn test_col_var() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let result = ops::col_var(x.view(), 0).unwrap();
        approx_eq(result[0], 2.0 / 3.0, DEFAULT_TOL);
        approx_eq(result[1], 8.0 / 3.0, DEFAULT_TOL);
    }
}

mod helper_functions {
    use super::*;

    #[test]
    fn test_add_intercept() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = ops::add_intercept(x.view()).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_vec_approx_eq(&result.column(0).to_vec(), &[1.0, 1.0], DEFAULT_TOL);
        approx_eq(result[[0, 1]], 1.0, DEFAULT_TOL);
        approx_eq(result[[0, 2]], 2.0, DEFAULT_TOL);
    }

    #[test]
    fn test_sigmoid() {
        approx_eq(ops::sigmoid(0.0), 0.5, DEFAULT_TOL);
        approx_eq(ops::sigmoid(f64::INFINITY), 1.0, 1e-6);
        approx_eq(ops::sigmoid(f64::NEG_INFINITY), 0.0, 1e-6);
    }

    #[test]
    fn test_softmax_mut() {
        let mut v = make_vector(&[1.0, 2.0, 3.0]);
        ops::softmax_mut(&mut v).unwrap();
        let sum: f64 = v.iter().sum();
        approx_eq(sum, 1.0, DEFAULT_TOL);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn test_argmax() {
        let v = make_vector(&[1.0, 5.0, 3.0, 2.0]);
        let result = ops::argmax(v.view()).unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_center_cols_mut() {
        let mut x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mean = make_vector(&[2.0, 3.0]);
        ops::center_cols_mut(&mut x, mean.view()).unwrap();
        assert_vec_approx_eq(&x.column(0).to_vec(), &[-1.0, 1.0], DEFAULT_TOL);
        assert_vec_approx_eq(&x.column(1).to_vec(), &[-1.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    fn test_scale_cols_mut() {
        let mut x = make_matrix(2, 2, &[2.0, 4.0, 4.0, 8.0]);
        let scale = make_vector(&[2.0, 4.0]);
        ops::scale_cols_mut(&mut x, scale.view()).unwrap();
        assert_vec_approx_eq(&x.column(0).to_vec(), &[1.0, 2.0], DEFAULT_TOL);
        assert_vec_approx_eq(&x.column(1).to_vec(), &[1.0, 2.0], DEFAULT_TOL);
    }
}

mod validation {
    use super::*;

    #[test]
    fn test_ensure_nonempty_vec_empty() {
        let v = Vector::zeros(0);
        let result = ops::ensure_nonempty_vec(v.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_ensure_nonempty_vec_ok() {
        let v = make_vector(&[1.0]);
        let result = ops::ensure_nonempty_vec(v.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_ensure_nonempty_mat_empty_rows() {
        let m = Matrix::zeros((0, 2));
        let result = ops::ensure_nonempty_mat(m.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_ensure_nonempty_mat_empty_cols() {
        let m = Matrix::zeros((2, 0));
        let result = ops::ensure_nonempty_mat(m.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_ensure_len_mismatch() {
        let a = make_vector(&[1.0, 2.0]);
        let b = make_vector(&[1.0, 2.0, 3.0]);
        let result = ops::ensure_len(a.view(), b.view(), "a", "b");
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_ensure_gemv_mismatch() {
        let a = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = make_vector(&[1.0, 2.0]);
        let result = ops::ensure_gemv(a.view(), x.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn test_dot_length_mismatch() {
        let a = make_vector(&[1.0, 2.0]);
        let b = make_vector(&[1.0, 2.0, 3.0]);
        let result = ops::dot(a.view(), b.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_matmul_dim_mismatch() {
        let a = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = ops::matmul(a.view(), b.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_col_var_invalid_ddof() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = ops::col_var(x.view(), 2);
        assert!(matches!(result, Err(Error::InvalidParam { .. })));
    }
}

#[cfg(feature = "faer-backend")]
mod solver {
    use super::*;

    #[test]
    fn test_solve_cholesky() {
        let a = make_matrix(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = make_vector(&[1.0, 2.0]);
        let result = ops::solve_cholesky(a.view(), b.view()).unwrap();
        let expected = [0.0909090909090909, 0.6363636363636364];
        assert_vec_approx_eq(&result.to_vec(), &expected, 1e-6);
    }

    #[test]
    fn test_solve_svd() {
        let a = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = make_vector(&[5.0, 11.0]);
        let result = ops::solve_svd(a.view(), b.view()).unwrap();
        assert_vec_approx_eq(&result.to_vec(), &[1.0, 2.0], 1e-6);
    }

    #[test]
    fn test_solve_cholesky_non_square() {
        let a = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_vector(&[1.0, 2.0]);
        let result = ops::solve_cholesky(a.view(), b.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }
}
