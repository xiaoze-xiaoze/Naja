use crate::core::{Result, Error};
use crate::core::compute::types::{MatrixView, VectorView};
use crate::core::compute::ops::{ensure_nonempty_mat, ensure_nonempty_vec};

pub fn silhouette_score(x: MatrixView<'_>, labels: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_mat(x)?;
    ensure_nonempty_vec(labels)?;
    if x.nrows() != labels.len() { return Err(Error::invalid_shape(format!("x.nrows()={}, labels.len()={}", x.nrows(), labels.len()))); }
    let n = x.nrows();
    if n < 2 { return Err(Error::invalid_shape("Silhouette Score requires at least 2 samples")); }
    let mut unique_labels: Vec<f64> = labels.iter().copied().collect();
    unique_labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_labels.dedup();
    if unique_labels.len() < 2 { return Err(Error::invalid_param("labels", "Silhouette Score requires at least 2 clusters")); }
    let mut s_sum = 0.0;
    for i in 0..n {
        let label_i = labels[i];
        let xi = x.row(i);
        let mut a_dist_sum = 0.0;
        let mut a_count = 0;
        let mut b_min_mean_dist = f64::MAX;
        for &label_k in &unique_labels {
            if label_k == label_i {
                for j in 0..n {
                    if i == j { continue; }
                    if labels[j] == label_i {
                        let diff = &xi - &x.row(j);
                        let d = diff.dot(&diff).sqrt();
                        a_dist_sum += d;
                        a_count += 1;
                    }
                }
            } 
            else {
                let mut dist_sum = 0.0;
                let mut count = 0;
                for j in 0..n {
                    if labels[j] == label_k {
                        let diff = &xi - &x.row(j);
                        let d = diff.dot(&diff).sqrt();
                        dist_sum += d;
                        count += 1;
                    }
                }
                if count > 0 {
                    let mean_dist = dist_sum / count as f64;
                    if mean_dist < b_min_mean_dist { b_min_mean_dist = mean_dist; }
                }
            }
        }
        let a = if a_count > 0 { a_dist_sum / a_count as f64 } else { 0.0 };
        let b = b_min_mean_dist;
        let s = if a == 0.0 && b == 0.0 { 0.0 } else { (b - a) / a.max(b) };
        s_sum += s;
    }
    Ok(s_sum / n as f64)
}
