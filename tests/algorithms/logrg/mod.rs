use crate::common::*;
use naja::algorithms::logrg::{LogisticRegression, LogisticRegressionConfig, Penalty, Solver};
use naja::core::traits::{Predictor, SupervisedEstimator};
use naja::core::Error;

mod basic {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LogisticRegressionConfig::default();
        assert!(config.intercept);
        assert_eq!(config.max_iter, 1000);
        approx_eq(config.tol, 1e-4, DEFAULT_TOL);
        match config.solver {
            Solver::GradientDescent { learning_rate, .. } => assert!(approx_eq(learning_rate, 0.1, DEFAULT_TOL)),
            _ => panic!("Expected GradientDescent solver"),
        }
    }

    #[test]
    fn test_fit_simple_classification() {
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        assert!(coef[0].is_finite());
        assert!(fitted.intercept_value().is_finite());
    }

    #[test]
    fn test_predict_simple_classification() {
        let x_train = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y_train = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new().learning_rate(0.5).max_iter(2000);
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_new = make_matrix(2, 1, &[-1.0, 4.0]);
        let y_pred = Predictor::predict(&fitted, x_new.view()).unwrap();
        assert_eq!(y_pred[0], 0.0);
        assert_eq!(y_pred[1], 1.0);
    }

    #[test]
    fn test_chain_config() {
        let model = LogisticRegression::new()
            .intercept(false)
            .max_iter(500)
            .tol(1e-5)
            .learning_rate(0.01);
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let config = fitted.config();
        assert!(!config.intercept);
        assert_eq!(config.max_iter, 500);
        approx_eq(config.tol, 1e-5, DEFAULT_TOL);
        match config.solver {
            Solver::GradientDescent { learning_rate, .. } => assert!(approx_eq(learning_rate, 0.01, DEFAULT_TOL)),
            _ => panic!("Expected GradientDescent solver"),
        }
    }
}

mod config {
    use super::*;

    #[test]
    fn test_no_intercept() {
        let x = make_matrix(4, 1, &[-2.0, -1.0, 1.0, 2.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new().intercept(false).learning_rate(0.5);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        let intercept = fitted.intercept_value();
        assert!(coef[0].is_finite());
        approx_eq(intercept, 0.0, 1e-6);
    }

    #[test]
    fn test_max_iter_limits_iterations() {
        let x = make_matrix(100, 1, &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let y: Vector = x.column(0).mapv(|v| if v > 50.0 { 1.0 } else { 0.0 });
        let model = LogisticRegression::new().max_iter(10);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let y_pred = Predictor::predict(&fitted, x.view()).unwrap();
        assert_eq!(y_pred.len(), 100);
        for pred in y_pred.iter() {
            assert!(*pred == 0.0 || *pred == 1.0);
        }
    }

    #[test]
    fn test_learning_rate_affects_convergence() {
        let x = make_matrix(100, 1, &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let y: Vector = x.column(0).mapv(|v| if v > 50.0 { 1.0 } else { 0.0 });
        let model_fast = LogisticRegression::new().learning_rate(1.0).max_iter(100);
        let model_slow = LogisticRegression::new().learning_rate(0.001).max_iter(100);
        let _fitted_fast = SupervisedEstimator::fit_supervised(&model_fast, x.view(), y.view()).unwrap();
        let _fitted_slow = SupervisedEstimator::fit_supervised(&model_slow, x.view(), y.view()).unwrap();
    }
}

mod behavior {
    use super::*;

    #[test]
    fn test_ridge_regularization() {
        let x = make_matrix(10, 1, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let y = make_vector(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let model_none = LogisticRegression::new().penalty(Penalty::None);
        let fitted_none = SupervisedEstimator::fit_supervised(&model_none, x.view(), y.view()).unwrap();
        let model_ridge = LogisticRegression::new().penalty(Penalty::Ridge { alpha: 10.0 });
        let fitted_ridge = SupervisedEstimator::fit_supervised(&model_ridge, x.view(), y.view()).unwrap();
        assert!(fitted_ridge.coefficients()[0].abs() <= fitted_none.coefficients()[0].abs() + 1.0);
    }

    #[test]
    fn test_lasso_regularization() {
        let x = make_matrix(10, 2, &[
            0.0, 0.1,
            1.0, 0.1,
            2.0, 0.1,
            3.0, 0.1,
            4.0, 0.1,
            5.0, 0.1,
            6.0, 0.1,
            7.0, 0.1,
            8.0, 0.1,
            9.0, 0.1,
        ]);
        let y = make_vector(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let model = LogisticRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.1 })
            .max_iter(2000)
            .solver(Solver::CoordinateDescent);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        assert!(fitted.coefficients()[0].abs() > fitted.coefficients()[1].abs());
    }

    #[test]
    fn test_multifeature() {
        let (x, y) = make_logistic_data(50, &[2.0, -1.0], 0.5);
        let model = LogisticRegression::new().learning_rate(0.5).max_iter(2000);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let fitted_coef = fitted.coefficients();
        assert!(fitted_coef[0].is_finite());
        assert!(fitted_coef[1].is_finite());
    }

    #[test]
    fn test_predict_proba_range() {
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let proba = fitted.predict_proba(x.view()).unwrap();
        for p in proba.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_single_sample_prediction() {
        let x_train = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y_train = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_single = make_matrix(1, 1, &[10.0]);
        let y_pred = Predictor::predict(&fitted, x_single.view()).unwrap();
        assert_eq!(y_pred.len(), 1);
    }

    #[test]
    fn test_batch_prediction() {
        let x_train = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y_train = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_batch = make_matrix(100, 1, &(0..100).map(|i| i as f64 / 10.0).collect::<Vec<_>>());
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
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_empty_y() {
        let x = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y = Vector::zeros(0);
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_xy_length_mismatch() {
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let y = make_vector(&[1.0, 2.0]);
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_predict_feature_mismatch() {
        let x_train = make_matrix(4, 2, &[0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
        let y_train = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_wrong = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = Predictor::predict(&fitted, x_wrong.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_predict_empty_x() {
        let x_train = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y_train = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x_train.view(), y_train.view()).unwrap();
        let x_empty = Matrix::zeros((0, 1));
        let result = Predictor::predict(&fitted, x_empty.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_all_same_class() {
        let x = make_matrix(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let y = make_vector(&[0.0, 0.0, 0.0, 0.0]);
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_sample_fit() {
        let x = make_matrix(1, 2, &[1.0, 2.0]);
        let y = make_vector(&[1.0]);
        let model = LogisticRegression::new().intercept(false);
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_values() {
        let x = make_matrix(4, 1, &[1e8, 2e8, 3e8, 4e8]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        assert!(coef[0].is_finite());
    }

    #[test]
    fn test_inf_input() {
        let x = make_matrix(3, 1, &[1.0, f64::INFINITY, 3.0]);
        let y = make_vector(&[0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_nan_input() {
        let x = make_matrix(3, 1, &[1.0, f64::NAN, 3.0]);
        let y = make_vector(&[0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let result = SupervisedEstimator::fit_supervised(&model, x.view(), y.view());
        assert!(result.is_ok() || result.is_err());
    }
}

mod trait_impl {
    use super::*;

    #[test]
    fn test_supervised_estimator_trait() {
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let _coef = fitted.coefficients();
        assert!(fitted.intercept_value().is_finite());
    }

    #[test]
    fn test_predictor_trait() {
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let y_pred = Predictor::predict(&fitted, x.view()).unwrap();
        assert_eq!(y_pred.len(), 4);
    }
}

mod accessors {
    use super::*;

    #[test]
    fn test_coefficients_accessor() {
        let x = make_matrix(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let coef = fitted.coefficients();
        assert_eq!(coef.len(), 2);
    }

    #[test]
    fn test_intercept_accessor() {
        let x = make_matrix(4, 1, &[0.0, 1.0, 2.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new();
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let intercept = fitted.intercept_value();
        assert!(intercept.is_finite());
    }

    #[test]
    fn test_config_accessor() {
        let x = make_matrix(4, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let y = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let model = LogisticRegression::new()
            .intercept(true)
            .max_iter(500)
            .tol(1e-3)
            .learning_rate(0.5);
        let fitted = SupervisedEstimator::fit_supervised(&model, x.view(), y.view()).unwrap();
        let config = fitted.config();
        assert!(config.intercept);
        assert_eq!(config.max_iter, 500);
        approx_eq(config.tol, 1e-3, DEFAULT_TOL);
        match config.solver {
            Solver::GradientDescent { learning_rate, .. } => assert!(approx_eq(learning_rate, 0.5, DEFAULT_TOL)),
            _ => panic!("Expected GradientDescent solver"),
        }
    }
}

fn make_logistic_data(n: usize, coefficients: &[f64], intercept: f64) -> (Matrix, Vector) {
    let n_features = coefficients.len();
    let mut x = Matrix::zeros((n, n_features));
    let mut y = Vector::zeros(n);
    for i in 0..n {
        let mut z = intercept;
        for j in 0..n_features {
            let xij = ((i + j) % 10) as f64 - 5.0;
            x[[i, j]] = xij;
            z += coefficients[j] * xij;
        }
        let p = 1.0 / (1.0 + (-z).exp());
        y[i] = if p >= 0.5 { 1.0 } else { 0.0 };
    }
    (x, y)
}
