use crate::core::compute::types::{MatrixView, VectorView};
use crate::core::regularization::Penalty;
use crate::core::traits::FitSupervised;
use crate::core::Result;
use super::fit;
use super::predict::LinearRegressionSolution;

#[derive(Debug, Clone)]
pub struct LinearRegressionConfig {
    pub intercept: bool,
    pub penalty: Penalty,
    pub max_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone)]
pub struct LinearRegression {
    config: LinearRegressionConfig,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self {
            intercept: true,
            penalty: Penalty::None,
            max_iter: 1000,
            tol: 1e-4,
        }
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self { config: LinearRegressionConfig::default() }
    }
}

impl LinearRegression {
    pub fn new() -> Self { Self::default() }
    pub fn config(mut self, config: LinearRegressionConfig) -> Self { self.config = config; self }
    pub fn intercept(mut self, intercept: bool) -> Self { self.config.intercept = intercept; self }
    pub fn penalty(mut self, penalty: Penalty) -> Self { self.config.penalty = penalty; self }
    pub fn max_iter(mut self, max_iter: usize) -> Self { self.config.max_iter = max_iter; self }
    pub fn tol(mut self, tol: f64) -> Self { self.config.tol = tol; self }
}

impl FitSupervised for LinearRegression {
    type Object = LinearRegressionSolution;
    fn fit(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Object> {
        fit::fit(&self.config, x, y)
    }
}
