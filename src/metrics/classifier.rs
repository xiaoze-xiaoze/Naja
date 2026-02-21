use crate::core::Result;
use crate::core::compute::types::{VectorView};
use crate::core::compute::ops::{ensure_len, ensure_nonempty_vec};

pub fn accuracy(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let n = y_true.len() as f64;
    let correct: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| if t == p { 1.0 } else { 0.0 }).sum();
    Ok(correct / n)
}

pub fn precision(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let mut tp = 0.0;
    let mut fp = 0.0;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if *p == pos_label {
            if *t == pos_label { tp += 1.0; }
            else { fp += 1.0; }
        }
    }
    if tp + fp == 0.0 { return Ok(0.0); }
    Ok(tp / (tp + fp))
}

pub fn recall(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let mut tp = 0.0;
    let mut fn_ = 0.0;
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if *t == pos_label {
            if *p == pos_label { tp += 1.0; }
            else { fn_ += 1.0; }
        }
    }
    if tp + fn_ == 0.0 { return Ok(0.0); }
    Ok(tp / (tp + fn_))
}

pub fn f1_score(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64> {
    let p = precision(y_true, y_pred, pos_label)?;
    let r = recall(y_true, y_pred, pos_label)?;
    if p + r == 0.0 { return Ok(0.0); }
    Ok(2.0 * p * r / (p + r))
}
