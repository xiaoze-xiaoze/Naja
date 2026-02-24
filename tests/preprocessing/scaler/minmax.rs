use crate::common::*;
use naja::core::traits::{FittableTransformer, InversibleTransformer, PartialFit, Transformer};
use naja::prelude::*;

mod basic {
    use super::*;

    #[test]
    fn test_fit_transform_default_range() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        let expected = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0];
        assert_vec_approx_eq(&result.iter().cloned().collect::<Vec<_>>(), &expected, 1e-10);
    }

    #[test]
    fn test_fit_transform_custom_range() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new().with_feature_range(-1.0, 1.0);
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        let expected = [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0];
        assert_vec_approx_eq(&result.iter().cloned().collect::<Vec<_>>(), &expected, 1e-10);
    }

    #[test]
    fn test_fit_simple_data() {
        let x = make_matrix(3, 1, &[0.0, 5.0, 10.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        let expected = [0.0, 0.5, 1.0];
        assert_vec_approx_eq(&result.column(0).to_vec(), &expected, 1e-10);
    }

    #[test]
    fn test_default() {
        let scaler = MinMaxScaler::default();
        let x = make_matrix(2, 1, &[0.0, 1.0]);
        let result = scaler.fit(x.view());
        assert!(result.is_ok());
    }
}

mod roundtrip {
    use super::*;

    #[test]
    fn test_inverse_transform() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = fitted.transform(x.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }

    #[test]
    fn test_inverse_transform_custom_range() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new().with_feature_range(-1.0, 1.0);
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = fitted.transform(x.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }

    #[test]
    fn test_inverse_transform_different_data() {
        let x_train = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x_train.view()).unwrap();
        let x_new = make_matrix(2, 2, &[2.0, 3.0, 4.0, 5.0]);
        let transformed = fitted.transform(x_new.view()).unwrap();
        let recovered = fitted.inverse_transform(transformed.view()).unwrap();
        assert!(matrix_all_close(&x_new, &recovered, 1e-10));
    }
}

mod partial_fit {
    use super::*;

    #[test]
    fn test_partial_fit_incremental() {
        let batch1 = make_matrix(2, 2, &[1.0, 10.0, 2.0, 20.0]);
        let batch2 = make_matrix(2, 2, &[0.0, 30.0, 3.0, 5.0]);
        let all_data = make_matrix(4, 2, &[1.0, 10.0, 2.0, 20.0, 0.0, 30.0, 3.0, 5.0]);
        let scaler_batch = MinMaxScaler::new();
        let fitted_batch = scaler_batch.fit(all_data.view()).unwrap();
        let scaler_incr = MinMaxScaler::new();
        let mut fitted_incr = scaler_incr.fit(batch1.view()).unwrap();
        fitted_incr.partial_fit(batch2.view(), None).unwrap();
        let result_batch = fitted_batch.transform(all_data.view()).unwrap();
        let result_incr = fitted_incr.transform(all_data.view()).unwrap();
        assert!(matrix_all_close(&result_batch, &result_incr, 1e-6));
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input_fit() {
        let x = Matrix::zeros((0, 2));
        let scaler = MinMaxScaler::new();
        let result = scaler.fit(x.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_transform() {
        let x_train = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x_train.view()).unwrap();
        let x_empty = Matrix::zeros((0, 2));
        let result = fitted.transform(x_empty.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch_transform() {
        let x_train = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x_train.view()).unwrap();
        let x_wrong = make_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = fitted.transform(x_wrong.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_constant_column() {
        let x = make_matrix(3, 2, &[5.0, 1.0, 5.0, 2.0, 5.0, 3.0]);
        let scaler = MinMaxScaler::new();
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
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        approx_eq(result[[0, 0]], 0.0, 1e-10);
        approx_eq(result[[0, 1]], 0.0, 1e-10);
    }

    #[test]
    fn test_negative_values() {
        let x = make_matrix(3, 1, &[-5.0, 0.0, 5.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        let expected = [0.0, 0.5, 1.0];
        assert_vec_approx_eq(&result.column(0).to_vec(), &expected, 1e-10);
    }
}

mod trait_impl {
    use super::*;

    #[test]
    fn test_fittable_transformer_trait() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = FittableTransformer::fit(scaler, x.view()).unwrap();
        let result = fitted.transform(x.view()).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_transformer_trait() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let result = Transformer::transform(&fitted, x.view()).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_inversible_transformer_trait() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scaler = MinMaxScaler::new();
        let fitted = scaler.fit(x.view()).unwrap();
        let transformed = <MinMaxScaler<Fitted> as Transformer>::transform(&fitted, x.view()).unwrap();
        let recovered = <MinMaxScaler<Fitted> as InversibleTransformer>::inverse_transform(&fitted, transformed.view()).unwrap();
        assert!(matrix_all_close(&x, &recovered, 1e-10));
    }

    #[test]
    fn test_partial_fit_trait() {
        let batch1 = make_matrix(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let batch2 = make_matrix(2, 2, &[3.0, 6.0, 4.0, 8.0]);
        let scaler = MinMaxScaler::new();
        let mut fitted = scaler.fit(batch1.view()).unwrap();
        PartialFit::partial_fit(&mut fitted, batch2.view(), None).unwrap();
    }
}
