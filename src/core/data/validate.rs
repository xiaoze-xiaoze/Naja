use crate::core::compute::types::{Matrix, Vector};
use crate::core::{Error, Result};

pub fn check_dimensions(records: &Matrix, targets: &Vector) -> Result<()> {
    if records.nrows() != targets.len() { return Err(Error::invalid_shape(format!("Number of records ({}) does not match number of targets ({})", records.nrows(), targets.len()))); }
    Ok(())
}

pub fn check_feature(n_features: usize, names: &[String]) -> Result<()> {
    if n_features != names.len() { return Err(Error::invalid_shape(format!("Number of features ({}) does not match number of names ({})", n_features, names.len()))); }
    Ok(())
}
