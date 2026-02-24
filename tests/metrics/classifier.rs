use crate::common::*;
use naja::metrics::classifier::{accuracy, f1_score, precision, recall};

mod correctness {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        let y_true = make_vector(&[1.0, 0.0, 1.0, 0.0]);
        let y_pred = make_vector(&[1.0, 0.0, 1.0, 0.0]);
        let result = accuracy(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_accuracy_half() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 0.0, 0.0, 1.0]);
        let result = accuracy(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.5, DEFAULT_TOL);
    }

    #[test]
    fn test_accuracy_zero() {
        let y_true = make_vector(&[1.0, 1.0, 1.0, 1.0]);
        let y_pred = make_vector(&[0.0, 0.0, 0.0, 0.0]);
        let result = accuracy(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_precision_perfect() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let result = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_precision_no_true_positives() {
        let y_true = make_vector(&[0.0, 0.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let result = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_precision_half() {
        let y_true = make_vector(&[1.0, 0.0, 1.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let result = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.5, DEFAULT_TOL);
    }

    #[test]
    fn test_recall_perfect() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let result = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_recall_no_positives_in_true() {
        let y_true = make_vector(&[0.0, 0.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 0.0, 1.0, 0.0]);
        let result = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }

    #[test]
    fn test_recall_half() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 0.0, 0.0, 0.0]);
        let result = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.5, DEFAULT_TOL);
    }

    #[test]
    fn test_f1_perfect() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let result = f1_score(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 1.0, DEFAULT_TOL);
    }

    #[test]
    fn test_f1_balanced() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 0.0, 1.0, 0.0]);
        let result = f1_score(y_true.view(), y_pred.view(), 1.0).unwrap();
        let p = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        let r = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        let expected = 2.0 * p * r / (p + r);
        approx_eq(result, expected, DEFAULT_TOL);
    }

    #[test]
    fn test_f1_zero_when_precision_recall_zero() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let result = f1_score(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }
}

mod edge_cases {
    use super::*;
    use naja::core::Error;

    #[test]
    fn test_accuracy_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = accuracy(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_accuracy_length_mismatch() {
        let y_true = make_vector(&[1.0, 0.0, 1.0]);
        let y_pred = make_vector(&[1.0, 0.0]);
        let result = accuracy(y_true.view(), y_pred.view());
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_precision_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = precision(y_true.view(), y_pred.view(), 1.0);
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_precision_length_mismatch() {
        let y_true = make_vector(&[1.0, 0.0, 1.0]);
        let y_pred = make_vector(&[1.0, 0.0]);
        let result = precision(y_true.view(), y_pred.view(), 1.0);
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_recall_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = recall(y_true.view(), y_pred.view(), 1.0);
        assert!(matches!(result, Err(Error::EmptyInput(_))));
    }

    #[test]
    fn test_recall_length_mismatch() {
        let y_true = make_vector(&[1.0, 0.0, 1.0]);
        let y_pred = make_vector(&[1.0, 0.0]);
        let result = recall(y_true.view(), y_pred.view(), 1.0);
        assert!(matches!(result, Err(Error::InvalidShape(_))));
    }

    #[test]
    fn test_f1_empty_input() {
        let y_true = Vector::zeros(0);
        let y_pred = Vector::zeros(0);
        let result = f1_score(y_true.view(), y_pred.view(), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_f1_length_mismatch() {
        let y_true = make_vector(&[1.0, 0.0, 1.0]);
        let y_pred = make_vector(&[1.0, 0.0]);
        let result = f1_score(y_true.view(), y_pred.view(), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_precision_no_predictions() {
        let y_true = make_vector(&[1.0, 1.0, 0.0, 0.0]);
        let y_pred = make_vector(&[0.0, 0.0, 0.0, 0.0]);
        let result = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        approx_eq(result, 0.0, DEFAULT_TOL);
    }
}

mod behavior {
    use super::*;

    #[test]
    fn test_multiclass_accuracy() {
        let y_true = make_vector(&[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let y_pred = make_vector(&[0.0, 1.0, 2.0, 1.0, 1.0, 0.0]);
        let result = accuracy(y_true.view(), y_pred.view()).unwrap();
        approx_eq(result, 4.0 / 6.0, DEFAULT_TOL);
    }

    #[test]
    fn test_binary_with_different_pos_label() {
        let y_true = make_vector(&[0.0, 0.0, 1.0, 1.0]);
        let y_pred = make_vector(&[0.0, 1.0, 1.0, 1.0]);
        let acc = accuracy(y_true.view(), y_pred.view()).unwrap();
        approx_eq(acc, 0.75, DEFAULT_TOL);
        let prec_neg = precision(y_true.view(), y_pred.view(), 0.0).unwrap();
        approx_eq(prec_neg, 1.0, DEFAULT_TOL);
        let rec_neg = recall(y_true.view(), y_pred.view(), 0.0).unwrap();
        approx_eq(rec_neg, 0.5, DEFAULT_TOL);
    }

    #[test]
    fn test_f1_formula() {
        let y_true = make_vector(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let p = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        let r = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        let f1 = f1_score(y_true.view(), y_pred.view(), 1.0).unwrap();
        let expected = 2.0 * p * r / (p + r);
        approx_eq(f1, expected, DEFAULT_TOL);
    }

    #[test]
    fn test_precision_recall_tradeoff() {
        let y_true = make_vector(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let y_pred = make_vector(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let p = precision(y_true.view(), y_pred.view(), 1.0).unwrap();
        let r = recall(y_true.view(), y_pred.view(), 1.0).unwrap();
        assert_eq!(p, 1.0);
        approx_eq(r, 0.5, DEFAULT_TOL);
    }
}
