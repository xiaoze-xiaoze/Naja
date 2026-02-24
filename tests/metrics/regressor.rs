use crate::common::*;
use naja::metrics::regressor::{mae, mse, r2_score, rmse, RegressionMetrics};

mod correctness {
    use super::*;

    #[test]
    fn test_mse_perfect_prediction() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = mse(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_mse_known_values() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[2.0, 3.0, 4.0]);
        let result = mse(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_rmse_perfect_prediction() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[1.0, 2.0, 3.0]);
        let result = rmse(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_rmse_known_values() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[2.0, 3.0, 4.0]);
        let result = rmse(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_mae_perfect_prediction() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[1.0, 2.0, 3.0]);
        let result = mae(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_mae_known_values() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[3.0, 4.0, 5.0]);
        let result = mae(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 2.0, DEFAULT_TOL);
    }

    #[test]
    fn test_r2_perfect_prediction() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_r2_mean_prediction() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[3.0, 3.0, 3.0, 3.0, 3.0]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, 1e-10);
    }

    #[test]
    fn test_r2_known_values() {
        let y_true = make_vector(&[3.0, 5.0, 7.0, 9.0, 11.0]);
        let y_pred = make_vector(&[2.5, 5.5, 6.5, 9.5, 10.5]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.975, 1e-6);
    }
}

mod regression_metrics_struct {
    use super::*;

    #[test]
    fn test_regression_metrics_new() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[1.1, 2.2, 2.9, 4.1, 4.8]);
        let metrics = RegressionMetrics::new(y_true.view(), y_pred.view()).unwrap();
        assert!(metrics.mse >= 0.0);
        assert!(metrics.rmse >= 0.0);
        assert!(metrics.mae >= 0.0);
        approx_eq(metrics.rmse, metrics.mse.sqrt(), DEFAULT_TOL);
    }

    #[test]
    fn test_regression_metrics_perfect() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = y_true.clone();
        let metrics = RegressionMetrics::new(y_true.view(), y_pred.view()).unwrap();
        approx_eq(metrics.mse, 0.0, DEFAULT_TOL);
        approx_eq(metrics.rmse, 0.0, DEFAULT_TOL);
        approx_eq(metrics.mae, 0.0, DEFAULT_TOL);
        approx_eq(metrics.r2, 1.0, DEFAULT_TOL);
    }
}

mod edge_cases {
    use super::*;
    use naja::core::Error;

    #[test]
    fn test_mse_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = mse(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_mse_length_mismatch() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[1.0, 2.0]);
        let result = mse(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_rmse_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = rmse(y_true.view(), y_pred.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_mae_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = mae(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_r2_single_sample() {
        let y_true = make_vector(&[1.0]);
        let y_pred = make_vector(&[1.0]);
        let result = r2_score(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_r2_constant_y_true() {
        let y_true = make_vector(&[5.0, 5.0, 5.0, 5.0]);
        let y_pred = make_vector(&[5.0, 5.0, 5.0, 5.0]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_r2_constant_y_true_bad_pred() {
        let y_true = make_vector(&[5.0, 5.0, 5.0, 5.0]);
        let y_pred = make_vector(&[1.0, 2.0, 3.0, 4.0]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_mae_length_mismatch() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[1.0, 2.0]);
        let result = mae(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_r2_length_mismatch() {
        let y_true = make_vector(&[1.0, 2.0, 3.0]);
        let y_pred = make_vector(&[1.0, 2.0]);
        let result = r2_score(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }
}

mod behavior {
    use super::*;

    #[test]
    fn test_mse_vs_mae_relationship() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[2.0, 3.0, 4.0, 5.0, 6.0]);
        let mse_val = mse(y_true.view(), y_pred.view()).unwrap();
        let mae_val = mae(y_true.view(), y_pred.view()).unwrap();
        assert!(mse_val >= mae_val * mae_val);
    }

    #[test]
    fn test_rmse_vs_mae() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[1.0, 3.0, 3.0, 5.0, 5.0]);
        let rmse_val = rmse(y_true.view(), y_pred.view()).unwrap();
        let mae_val = mae(y_true.view(), y_pred.view()).unwrap();
        assert!(rmse_val >= mae_val);
    }

    #[test]
    fn test_negative_r2_possible() {
        let y_true = make_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = make_vector(&[10.0, 10.0, 10.0, 10.0, 10.0]);
        let result = r2_score(y_true.view(), y_pred.view()).unwrap();
        assert!(result < 0.0);
    }
}
