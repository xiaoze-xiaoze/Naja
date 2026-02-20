use crate::core::compute::types::{Matrix, Vector};
use crate::core::Result;
use super::validate;

#[derive(Debug, Clone)]
pub struct Dataset {
    pub records: Matrix,
    pub targets: Vector,
    pub feature_names: Vec<String>,
}

impl Dataset {
    pub fn new(records: Matrix, targets: Vector) -> Result<Self> {
        validate::check_dimensions(&records, &targets)?;
        let feature_names = (0..records.ncols()).map(|i| format!("feature_{}", i)).collect();
        Ok(Self { records, targets, feature_names })
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        validate::check_feature(self.records.ncols(), &names)?;
        self.feature_names = names;
        Ok(self)
    }

    pub fn split(&self, test_ratio: f64) -> (Self, Self) {
        let n = self.records.nrows();
        let n_test = (n as f64 * test_ratio).round() as usize;
        let n_train = n - n_test;
        let train_records = self.records.slice(ndarray::s![..n_train, ..]).to_owned();
        let test_records = self.records.slice(ndarray::s![n_train.., ..]).to_owned();
        let train_targets = self.targets.slice(ndarray::s![..n_train]).to_owned();
        let test_targets = self.targets.slice(ndarray::s![n_train..]).to_owned();
        let train = Dataset {
            records: train_records,
            targets: train_targets,
            feature_names: self.feature_names.clone()
        };
        let test = Dataset {
            records: test_records,
            targets: test_targets,
            feature_names: self.feature_names.clone()
        };
        (train, test)
    }
}
