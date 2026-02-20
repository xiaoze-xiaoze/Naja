use crate::core::{Result, Error};
use crate::core::compute::types::{VectorView};
use crate::core::compute::ops::{ensure_len, ensure_nonempty_vec, l2_sq};

pub fn mse(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let n = y_true.len() as f64;
    let ss: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| { let d = t - p; d * d }).sum();
    Ok(ss / n)
}

pub fn rmse(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64> { Ok(mse(y_true, y_pred)?.sqrt()) }

pub fn mae(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let n = y_true.len() as f64;
    let sa: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).abs()).sum();
    Ok(sa / n)
}

pub fn r2_score(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(y_true)?;
    ensure_len(y_true, y_pred, "y_true", "y_pred")?;
    let n = y_true.len();
    if n < 2 { return Err(Error::invalid_shape("R2 score requires at least 2 samples")); }
    let y_mean = y_true.mean().unwrap_or(0.0);
    let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| { let d = t - p; d * d }).sum();
    let ss_tot: f64 = y_true.iter().map(|t| { let d = t - y_mean; d * d }).sum();
    if ss_tot == 0.0 {
        if ss_res == 0.0 { return Ok(1.0); }
        return Ok(0.0);
    }
    Ok(1.0 - (ss_res / ss_tot))
}
