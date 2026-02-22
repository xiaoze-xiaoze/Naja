use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};
use crate::core::Result;
use super::validate;

#[derive(Debug)]
pub struct Dataset {
    pub records: Matrix,
    pub targets: Vector,
    pub feature_names: Vec<String>
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetView<'a> {
    pub records: MatrixView<'a>,
    pub targets: VectorView<'a>,
    pub feature_names: &'a [String]
}

impl Dataset {
    pub fn new(records: Matrix, targets: Vector) -> Result<Self> {
        validate::check_dimensions(records.view(), targets.view())?;
        let feature_names = (0..records.ncols()).map(|i| format!("feature_{}", i)).collect();
        Ok(Self { records, targets, feature_names })
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Result<Self> {
        validate::check_feature(self.records.ncols(), &names)?;
        self.feature_names = names;
        Ok(self)
    }

    pub fn as_view(&self) -> DatasetView<'_> {
        DatasetView {
            records: self.records.view(),
            targets: self.targets.view(),
            feature_names: &self.feature_names
        }
    }

    pub fn split(&self, test_ratio: f64) -> Result<(DatasetView<'_>, DatasetView<'_>)> {
        validate::check_split_ratio(test_ratio)?;
        let n = self.records.nrows();
        let n_test = (n as f64 * test_ratio).round() as usize;
        let n_train = n - n_test;
        let train = DatasetView {
            records: self.records.slice(ndarray::s![..n_train, ..]),
            targets: self.targets.slice(ndarray::s![..n_train]),
            feature_names: &self.feature_names
        };
        let test = DatasetView {
            records: self.records.slice(ndarray::s![n_train.., ..]),
            targets: self.targets.slice(ndarray::s![n_train..]),
            feature_names: &self.feature_names
        };
        Ok((train, test))
    }

    pub fn nrows(&self) -> usize { self.records.nrows() }
    pub fn ncols(&self) -> usize { self.records.ncols() }
}

impl<'a> DatasetView<'a> {
    pub fn to_owned(&self) -> Dataset {
        Dataset {
            records: self.records.to_owned(),
            targets: self.targets.to_owned(),
            feature_names: self.feature_names.to_vec()
        }
    }
    pub fn nrows(&self) -> usize { self.records.nrows() }
    pub fn ncols(&self) -> usize { self.records.ncols() }
}
