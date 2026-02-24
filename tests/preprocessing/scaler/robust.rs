use crate::common::*;
use naja::core::traits::{FittableTransformer, InversibleTransformer, Transformer};
use naja::prelude::*;

mod basic {
    use super::*;

    #[test]
    fn test_fit_transform_default() {
        let x = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[5, 1]);
        approx_eq(result[[2, 0]], 0.0, 1e-10);
    }

    #[test]
    fn test_fit_with_custom_quantile_range() {
        let x = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaler = RobustScaler::new().with_quantile_range(0.1, 0.9);
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[5, 1]);
    }

    #[test]
    fn test_fit_center_only() {
        let x = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaler = RobustScaler::new().with_center(true).with_scale(false);
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        approx_eq(result[[2, 0]], 0.0, 1e-10);
    }

    #[test]
    fn test_fit_scale_only() {
        let x = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaler = RobustScaler::new().with_center(false).with_scale(true);
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[5, 1]);
    }

    #[test]
    fn test_default() {
        let scaler = RobustScaler::default();
        let x = make_matrix(2, 1, &[0.0, 1.0]);
        let result = scaler.fit(x.view());
        assert!(result.is_ok());
    }
}

mod roundtrip {
    use super::*;

    #[test]
    fn test_inverse_transform() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = fitted.transform(x.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }

    #[test]
    fn test_inverse_transform_center_only() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new().with_center(true).with_scale(false);
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = fitted.transform(x.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }

    #[test]
    fn test_inverse_transform_scale_only() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new().with_center(false).with_scale(true);
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = fitted.transform(x.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }
}

mod behavior {
    use super::*;

    #[test]
    fn test_outlier_robustness() {
        let x_with_outlier = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 100.0]);
        let x_normal = make_matrix(5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let scaler_with = RobustScaler::new();
        let fitted_with = scaler_with.fit(x_with_outlier.view()).unwrap();
        let scaler_normal = RobustScaler::new();
        let fitted_normal = scaler_normal.fit(x_normal.view()).unwrap();
        let result_with = fitted_with.transform(x_with_outlier.view()).unwrap();
        let result_normal = fitted_normal.transform(x_normal.view()).unwrap();
        assert!((result_with[[2, 0]] - result_normal[[2, 0]]).abs() < 1.0);
    }

    #[test]
    fn test_multicolumn() {
        let x = make_matrix(4, 3, &[1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[4, 3]);
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input_fit() {
        let x = Matrix::zeros((0, 2));
        let scaler = RobustScaler::new();
        let result = scaler.fit(x.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_transform() {
        let x_train = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x_train.view()).unwrap();
        let x_empty = Matrix::zeros((0, 2));
        let result = fitted.transform(x_empty.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_transform() {
        let x_train = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x_train.view()).unwrap();
        let x_wrong = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = fitted.transform(x_wrong.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_column() {
        let x = make_matrix(4, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        let col0 = result.column(0);
        for v in col0.iter() {
            approx_eq(*v, 0.0, 1e-10);
        }
    }

    #[test]
    fn test_single_sample() {
        let x = make_matrix(1, 2, &[1.0, 2.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        approx_eq(result[[0, 0]], 0.0, 1e-10);
        approx_eq(result[[0, 1]], 0.0, 1e-10);
    }
}

mod trait_impl {
    use super::*;

    #[test]
    fn test_fittable_transformer_trait() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new();
        let fitted = FittableTransformer::fit(scaler, x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[5, 2]);
    }

    #[test]
    fn test_transformer_trait() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = Transformer::transform(&fitted, x.view()).unwrap();
        assert_eq!(result.shape(), &[5, 2]);
    }

    #[test]
    fn test_inversible_transformer_trait() {
        let x = make_matrix(5, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
        let scaler = RobustScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = <RobustScaler<Fitted> as Transformer>::transform(&fitted, x.view()).unwrap();
        let recovered = <RobustScaler<Fitted> as InversibleTransformer>::inverse_transform(&fitted, transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }
}
