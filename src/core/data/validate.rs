use crate::core::compute::types::{MatrixView, VectorView};
use crate::core::{Error, Result};

pub fn check_dimensions(records: MatrixView<'_>, targets: VectorView<'_>) -> Result<()> {
    if records.nrows() != targets.len() { return Err(Error::invalid_shape(format!("Number of records ({}) does not match number of targets ({})", records.nrows(), targets.len()))); }
    Ok(())
}

pub fn check_feature(n_features: usize, names: &[String]) -> Result<()> {
    if n_features != names.len() { return Err(Error::invalid_shape(format!("Number of features ({}) does not match number of names ({})", n_features, names.len()))); }
    Ok(())
}

pub fn check_split_ratio(ratio: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&ratio) { return Err(Error::invalid_param("test_ratio", &format!("must be in [0.0, 1.0], got {}", ratio))); }
    Ok(())
}
