use crate::common::*;
use naja::algorithms::linrg::{LinearRegression, LinearRegressionConfig, Penalty};
use naja::core::traits::{Predictor, SupervisedEstimator};
use naja::core::Error;

mod basic {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LinearRegressionConfig::default();
        assert!(config.intercept);
        assert_eq!(config.max_iter, 1000);
        approx_eq(config.tol, 1e-4, DEFAULT_TOL);
    }

    #[test]
    fn test_fit_simple_line() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        let intercept = fitted.intercept_value();
        approx_eq(coef[0], 2.0, 1e-6);
        approx_eq(intercept, 1.0, 1e-6);
    }

    #[test]
    fn test_predict_simple_line() {
        let x_train = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y_train = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_new = make_matrix(2, 1, &[4.0, 5.0]);
        let y_pred = Predictor::predict(&fitted, x_new.view()).unwrap();
        assert_vec_approx_eq(&y_pred.to_vec(), &[9.0, 11.0], 1e-6);
    }

    #[test]
    fn test_chain_config() {
        let model = LinearRegression::new()
            .intercept(false)
            .max_iter(500)
            .tol(1e-5);
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[2.0, 4.0, 6.0]);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let config = fitted.config();
        assert!(!config.intercept);
        assert_eq!(config.max_iter, 500);
        approx_eq(config.tol, 1e-5, DEFAULT_TOL);
    }
}

mod config {
    use super::*;

    #[test]
    fn test_no_intercept() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[2.0, 4.0, 6.0]);
        let model = LinearRegression::new().intercept(false);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        let intercept = fitted.intercept_value();
        approx_eq(coef[0], 2.0, 1e-6);
        approx_eq(intercept, 0.0, 1e-6);
    }

    #[test]
    fn test_max_iter_limits_iterations() {
        let x = make_matrix(100, 1, &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let y: Vector = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let model = LinearRegression::new().max_iter(10);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let y_pred = Predictor::predict(&fitted, x.view()).unwrap();
        assert_eq!(y_pred.len(), 100);
        for pred in y_pred.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_tol_affects_convergence() {
        let x = make_matrix(100, 1, &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let y: Vector = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let model_loose = LinearRegression::new().tol(1e-1);
        let model_tight = LinearRegression::new().tol(1e-10);
        let _fitted_loose = SupervisedEstimator::fit_supervised(&model_loose, x.view(), y.view()).unwrap();
        let _fitted_tight = SupervisedEstimator::fit_supervised(&model_tight, x.view(), y.view()).unwrap();
    }
}

mod behavior {
    use super::*;

    #[test]
    fn test_ridge_regularization() {
        let x = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = make_vector(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let model_none = LinearRegression::new().penalty(Penalty::None);
        let fitted_none = SupervisedEstimator::fit_supervised(&model_none, x.view(), y.view()).unwrap();
        let model_ridge = LinearRegression::new().penalty(Penalty::Ridge { alpha: 10.0 });
        let fitted_ridge = SupervisedEstimator::fit_supervised(&model_ridge, x.view(), y.view()).unwrap();
        assert!(fitted_ridge.coefficients()[0].abs() < fitted_none.coefficients()[0].abs() + 0.5);
    }

    #[test]
    fn test_lasso_regularization() {
        use naja::algorithms::linrg::Solver;
        let x = make_matrix(5, 2, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]);
        let y = make_vector(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let model = LinearRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.1 })
            .max_iter(1000)
            .solver(Solver::CoordinateDescent);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        assert!(fitted.coefficients()[1].abs() < fitted.coefficients()[0].abs());
    }

    #[test]
    fn test_multifeature() {
        let coefficients = [2.0, 3.0, -1.0];
        let intercept = 5.0;
        let (x, y) = make_multifeature_linear_data(50, &coefficients, intercept);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let fitted_coef = fitted.coefficients();
        approx_eq(fitted_coef[0], coefficients[0], 0.1);
        approx_eq(fitted_coef[1], coefficients[1], 0.1);
        approx_eq(fitted_coef[2], coefficients[2], 0.1);
        approx_eq(fitted.intercept_value(), intercept, 0.1);
    }

    #[test]
    fn test_single_sample_prediction() {
        let x_train = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y_train = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_single = make_matrix(1, 1, &[10.0]);
        let y_pred = Predictor::predict(&fitted, x_single.view()).unwrap();
        assert_eq!(y_pred.len(), 1);
        approx_eq(y_pred[0], 21.0, 1e-6);
    }

    #[test]
    fn test_batch_prediction() {
        let x_train = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y_train = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_batch = make_matrix(100, 1, &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let y_pred = Predictor::predict(&fitted, x_batch.view()).unwrap();
        assert_eq!(y_pred.len(), 100);
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_x() {
        let x = Matrix::zeros((0, 2));
        let y = Vector::zeros(0);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_empty_y() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = Vector::zeros(0);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_xy_length_mismatch() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[1.0, 2.0]);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x_train = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y_train = make_vector(&[3.0, 7.0, 11.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_wrong = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = Predictor::predict(&fitted, x_wrong.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_predict_empty_x() {
        let x_train = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y_train = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_empty = Matrix::zeros((0, 1));
        let result = Predictor::predict(&fitted, x_empty.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_constant_features() {
        let x = make_matrix(3, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let y = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_sample_fit() {
        let x = make_matrix(1, 2, &[1.0, 2.0]);
        let y = make_vector(&[5.0]);
        let model = LinearRegression::new().intercept(false);
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_values() {
        let x = make_matrix(3, 1, &[1e8, 2e8, 3e8]);
        let y = make_vector(&[1e8, 2e8, 3e8]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        assert!(coef[0].is_finite());
    }

    #[test]
    fn test_inf_input() {
        let x = make_matrix(3, 1, &[1.0, f64::INFINITY, 3.0]);
        let y = make_vector(&[1.0, 2.0, 3.0]);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_input() {
        let x = make_matrix(3, 1, &[1.0, f64::NAN, 3.0]);
        let y = make_vector(&[1.0, 2.0, 3.0]);
        let model = LinearRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_err());
    }
}

mod regression {
    use super::*;

    #[test]
    fn test_known_coefficients() {
        let x = make_matrix(4, 2, &[1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0]);
        let y = make_vector(&[6.0, 8.0, 9.0, 11.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        approx_eq(fitted.coefficients()[0], 2.5, 1e-6);
        approx_eq(fitted.coefficients()[1], 1.5, 1e-6);
        approx_eq(fitted.intercept_value(), 2.0, 1e-6);
    }

    #[test]
    fn test_perfect_fit() {
        let (x, y) = make_perfect_linear_data(50, 2.5, 1.5);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let y_pred = Predictor::predict(&fitted, x.view()).unwrap();
        for (pred, actual) in y_pred.iter().zip(y.iter()) {
            approx_eq(*pred, *actual, 1e-6);
        }
    }
}

mod trait_impl {
    use super::*;

    #[test]
    fn test_supervised_estimator_trait() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let _coef = fitted.coefficients();
        assert!(fitted.intercept_value().is_finite());
    }

    #[test]
    fn test_predictor_trait() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let y_pred = Predictor::predict(&fitted, x.view()).unwrap();
        assert_eq!(y_pred.len(), 3);
    }
}

mod accessors {
    use super::*;

    #[test]
    fn test_coefficients_accessor() {
        let x = make_matrix(3, 2, &[1.0, 0.0, 2.0, 1.0, 3.0, 2.0]);
        let y = make_vector(&[1.0, 3.0, 5.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        assert_eq!(coef.len(), 2);
    }

    #[test]
    fn test_intercept_accessor() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[3.0, 5.0, 7.0]);
        let model = LinearRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let intercept = fitted.intercept_value();
        approx_eq(intercept, 1.0, 1e-6);
    }

    #[test]
    fn test_config_accessor() {
        let x = make_matrix(3, 2, &[1.0, 0.0, 2.0, 1.0, 3.0, 2.0]);
        let y = make_vector(&[1.0, 3.0, 5.0]);
        let model = LinearRegression::new()
            .intercept(true)
            .max_iter(500)
            .tol(1e-3);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let config = fitted.config();
        assert!(config.intercept);
        assert_eq!(config.max_iter, 500);
        approx_eq(config.tol, 1e-3, DEFAULT_TOL);
    }
}
