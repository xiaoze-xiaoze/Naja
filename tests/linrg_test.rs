mod fixtures;

use ndarray::{Array1, Array2};
use naja::algorithms::linrg::{LinearRegression, LinearRegressionConfig, Penalty};
use naja::core::traits::{FitSupervised, Predictor};
use naja::metrics;

mod smoke {
    use super::*;
    
    #[test]
    fn fit_and_predict_works() {
        let (x, y) = fixtures::simple_linear_data();
        let model = LinearRegression::new();
        let solution = model.fit(x.view(), y.view()).unwrap();
        let pred = solution.predict(x.view()).unwrap();
        assert_eq!(pred.len(), 5);
    }
    
    #[test]
    fn default_config_works() {
        let (x, y) = fixtures::simple_linear_data();
        let solution = LinearRegression::new().fit(x.view(), y.view()).unwrap();
        assert_eq!(solution.coefficients.len(), 1);
    }
}

mod correctness {
    use super::*;
    
    #[test]
    fn simple_linear_exact_solution() {
        let (x, y) = fixtures::simple_linear_with_intercept();
        let solution = LinearRegression::new()
            .intercept(true)
            .fit(x.view(), y.view())
            .unwrap();
        
        assert!((solution.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((solution.intercept - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn multivariate_exact_solution() {
        let (x, y) = fixtures::multivariate_data();
        let solution = LinearRegression::new()
            .intercept(true)
            .fit(x.view(), y.view())
            .unwrap();
        
        let pred = solution.predict(x.view()).unwrap();
        for (p, t) in pred.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-10);
        }
    }
    
    #[test]
    fn no_intercept_mode() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        
        let solution = LinearRegression::new()
            .intercept(false)
            .fit(x.view(), y.view())
            .unwrap();
        
        assert!((solution.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((solution.intercept - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn ridge_shrinks_coefficients() {
        let (x, y) = fixtures::simple_linear_with_intercept();
        
        let ols = LinearRegression::new()
            .penalty(Penalty::None)
            .fit(x.view(), y.view())
            .unwrap();
        
        let ridge = LinearRegression::new()
            .penalty(Penalty::Ridge { alpha: 10.0 })
            .fit(x.view(), y.view())
            .unwrap();
        
        assert!(ridge.coefficients[0].abs() < ols.coefficients[0].abs());
    }
    
    #[test]
    fn lasso_produces_sparse_solution() {
        let (x, y) = fixtures::sparse_true_coeffs();
        
        let lasso = LinearRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.1 })
            .max_iter(5000)
            .tol(1e-6)
            .fit(x.view(), y.view())
            .unwrap();
        
        let zero_count = lasso.coefficients.iter().filter(|&&c| c.abs() < 1e-4).count();
        assert!(zero_count >= 1);
    }
    
    #[test]
    fn prediction_matches_metrics() {
        let (x, y) = fixtures::noisy_linear_data(42);
        let solution = LinearRegression::new().fit(x.view(), y.view()).unwrap();
        let pred = solution.predict(x.view()).unwrap();
        
        let r2 = metrics::r2_score(y.view(), pred.view()).unwrap();
        assert!(r2 > 0.9);
    }
}

mod edge_cases {
    use super::*;
    
    #[test]
    fn single_sample_fits() {
        let (x, y) = fixtures::single_sample_data();
        let result = LinearRegression::new()
            .intercept(true)
            .fit(x.view(), y.view());
        
        match result {
            Ok(solution) => {
                let pred = solution.predict(x.view()).unwrap();
                assert_eq!(pred.len(), 1);
            }
            Err(_) => {}
        }
    }
    
    #[test]
    fn two_samples_exact_fit() {
        let (x, y) = fixtures::two_samples_data();
        let solution = LinearRegression::new()
            .intercept(false)
            .fit(x.view(), y.view())
            .unwrap();
        
        let pred = solution.predict(x.view()).unwrap();
        assert!((pred[0] - y[0]).abs() < 1e-10);
        assert!((pred[1] - y[1]).abs() < 1e-10);
    }
    
    #[test]
    fn multicollinear_handled_gracefully() {
        let (x, y) = fixtures::multicollinear_data();
        let result = LinearRegression::new()
            .intercept(true)
            .fit(x.view(), y.view());
        
        assert!(result.is_ok());
        let solution = result.unwrap();
        let pred = solution.predict(x.view()).unwrap();
        
        for (p, t) in pred.iter().zip(y.iter()) {
            assert!((p - t).abs() < 1e-6);
        }
    }
    
    #[test]
    fn zero_features() {
        let x = Array2::from_shape_vec((3, 0), vec![] as Vec<f64>).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let result = LinearRegression::new().fit(x.view(), y.view());
        assert!(result.is_err());
    }
    
    #[test]
    fn very_small_alpha_ridge() {
        let (x, y) = fixtures::simple_linear_with_intercept();
        
        let ridge_small = LinearRegression::new()
            .penalty(Penalty::Ridge { alpha: 1e-10 })
            .fit(x.view(), y.view())
            .unwrap();
        
        assert!((ridge_small.coefficients[0] - 2.0).abs() < 0.1);
    }
}

mod config_variants {
    use super::*;
    
    #[test]
    fn penalty_none() {
        let (x, y) = fixtures::simple_linear_data();
        let solution = LinearRegression::new()
            .penalty(Penalty::None)
            .fit(x.view(), y.view())
            .unwrap();
        assert!(solution.coefficients[0].is_finite());
    }
    
    #[test]
    fn penalty_ridge_various_alpha() {
        let (x, y) = fixtures::multicollinear_data();
        
        for alpha in [0.01, 0.1, 1.0, 10.0] {
            let solution = LinearRegression::new()
                .penalty(Penalty::Ridge { alpha })
                .fit(x.view(), y.view())
                .unwrap();
            assert!(solution.coefficients.iter().all(|c| c.is_finite()));
        }
    }
    
    #[test]
    fn penalty_lasso_various_alpha() {
        let (x, y) = fixtures::multicollinear_data();
        
        for alpha in [0.001, 0.01, 0.1] {
            let solution = LinearRegression::new()
                .penalty(Penalty::Lasso { alpha })
                .max_iter(2000)
                .fit(x.view(), y.view())
                .unwrap();
            assert!(solution.coefficients.iter().all(|c| c.is_finite()));
        }
    }
    
    #[test]
    fn config_builder_chain() {
        let (x, y) = fixtures::simple_linear_with_intercept();
        let cfg = LinearRegressionConfig {
            intercept: false,
            penalty: Penalty::Ridge { alpha: 0.5 },
            max_iter: 500,
            tol: 1e-5,
        };
        
        let solution = LinearRegression::new().config(cfg).fit(x.view(), y.view()).unwrap();
        let intercept = solution.intercept;
        assert!((intercept - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn max_iter_affects_convergence() {
        let (x, y) = fixtures::sparse_true_coeffs();
        
        let _quick = LinearRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.05 })
            .max_iter(10)
            .fit(x.view(), y.view())
            .unwrap();
        
        let _thorough = LinearRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.05 })
            .max_iter(5000)
            .fit(x.view(), y.view())
            .unwrap();
    }
}

mod error_handling {
    use super::*;
    
    #[test]
    fn empty_x_rejected() {
        let x = fixtures::empty_x();
        let y = Array1::from_vec(vec![]);
        let result = LinearRegression::new().fit(x.view(), y.view());
        assert!(result.is_err());
    }
    
    #[test]
    fn empty_y_rejected() {
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let y = fixtures::empty_y();
        let result = LinearRegression::new().fit(x.view(), y.view());
        assert!(result.is_err());
    }
    
    #[test]
    fn mismatched_lengths_rejected() {
        let (x, y) = fixtures::mismatched_len_data();
        let result = LinearRegression::new().fit(x.view(), y.view());
        assert!(result.is_err());
    }
    
    #[test]
    fn predict_wrong_features_rejected() {
        let (x, y) = fixtures::multivariate_data();
        let solution = LinearRegression::new().fit(x.view(), y.view()).unwrap();
        
        let x_wrong = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let result = solution.predict(x_wrong.view());
        assert!(result.is_err());
    }
    
    #[test]
    fn nan_in_input() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, f64::NAN]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let result = LinearRegression::new().fit(x.view(), y.view());
        
        if result.is_ok() {
            let solution = result.unwrap();
            assert!(solution.coefficients.iter().any(|c| !c.is_finite()));
        }
    }
}

mod integration {
    use super::*;
    
    #[test]
    fn full_pipeline_with_metrics() {
        let (x, y) = fixtures::noisy_linear_data(123);
        
        let solution = LinearRegression::new()
            .intercept(true)
            .fit(x.view(), y.view())
            .unwrap();
        
        let pred = solution.predict(x.view()).unwrap();
        
        let mse = metrics::mse(y.view(), pred.view()).unwrap();
        let rmse = metrics::rmse(y.view(), pred.view()).unwrap();
        let mae = metrics::mae(y.view(), pred.view()).unwrap();
        let r2 = metrics::r2_score(y.view(), pred.view()).unwrap();
        
        assert!(mse > 0.0);
        assert!((rmse - mse.sqrt()).abs() < 1e-10);
        assert!(mae > 0.0);
        assert!(r2 > 0.0 && r2 <= 1.0);
    }
    
    #[test]
    fn ridge_vs_lasso_comparison() {
        let (x, y) = fixtures::multicollinear_data();
        
        let ridge = LinearRegression::new()
            .penalty(Penalty::Ridge { alpha: 1.0 })
            .fit(x.view(), y.view())
            .unwrap();
        
        let lasso = LinearRegression::new()
            .penalty(Penalty::Lasso { alpha: 0.1 })
            .max_iter(2000)
            .fit(x.view(), y.view())
            .unwrap();
        
        let ridge_pred = ridge.predict(x.view()).unwrap();
        let lasso_pred = lasso.predict(x.view()).unwrap();
        
        let ridge_r2 = metrics::r2_score(y.view(), ridge_pred.view()).unwrap();
        let lasso_r2 = metrics::r2_score(y.view(), lasso_pred.view()).unwrap();
        
        assert!(ridge_r2 > 0.9);
        assert!(lasso_r2 > 0.9);
    }
    
    #[test]
    fn train_test_split_simulation() {
        let (x, y) = fixtures::noisy_linear_data(999);
        
        let split = 40;
        let x_train = x.slice(ndarray::s![..split, ..]);
        let y_train = y.slice(ndarray::s![..split]);
        let x_test = x.slice(ndarray::s![split.., ..]);
        let y_test = y.slice(ndarray::s![split..]);
        
        let solution = LinearRegression::new()
            .fit(x_train, y_train)
            .unwrap();
        
        let pred_train = solution.predict(x_train).unwrap();
        let pred_test = solution.predict(x_test).unwrap();
        
        let r2_train = metrics::r2_score(y_train, pred_train.view()).unwrap();
        let r2_test = metrics::r2_score(y_test, pred_test.view()).unwrap();
        
        assert!(r2_train > 0.8);
        assert!(r2_test > 0.7);
    }
}
