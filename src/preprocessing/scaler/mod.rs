pub mod standard;
pub mod minmax;
pub mod robust;

pub use standard::StandardScaler;
pub use minmax::MinMaxScaler;
pub use robust::RobustScaler;

fn column_quantile(col: ndarray::ArrayView1<'_, f64>, q: f64) -> f64 {
    let mut sorted = col.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let pos = (q * (n - 1) as f64) as usize;
    let frac = q * (n - 1) as f64 - pos as f64;
    if pos + 1 < n {
        sorted[pos] * (1.0 - frac) + sorted[pos + 1] * frac
    } else {
        sorted[pos]
    }
}
