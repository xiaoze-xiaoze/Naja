use std::marker::PhantomData;
use crate::core::{Result, Error};
use crate::core::compute::types::{Vector, VectorView, MatrixView};
use crate::core::compute::ops;
use crate::core::traits::{Component, State, Unfitted, Fitted, Predictor, SupervisedEstimator};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Penalty {
    None,
    Ridge { alpha: f64 },
    Lasso { alpha: f64 },
}

impl Default for Penalty {
    fn default() -> Self { Penalty::None }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Solver {
    GradientDescent { learning_rate: f64, grad_tol: f64 },
    Irls,
    Sgd { batch_size: usize, learning_rate: f64 },
    Lbfgs { memory_size: usize },
    CoordinateDescent,
}

impl Default for Solver {
    fn default() -> Self { 
        Solver::GradientDescent { learning_rate: 0.1, grad_tol: 1e-6 } 
    }
}

#[derive(Debug, Clone)]
pub struct LogisticRegressionConfig {
    pub intercept: bool,
    pub penalty: Penalty,
    pub max_iter: usize,
    pub tol: f64,
    pub solver: Solver,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self { intercept: true, penalty: Penalty::None, max_iter: 1000, tol: 1e-4, solver: Solver::default() }
    }
}

#[derive(Debug, Clone)]
pub struct LogisticRegression<S: State = Unfitted> {
    config: LogisticRegressionConfig,
    coefficients: Option<Vector>,
    intercept_value: Option<f64>,
    _state: PhantomData<S>,
}

impl Default for LogisticRegression<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl LogisticRegression<Unfitted> {
    pub fn new() -> Self {
        Self {
            config: LogisticRegressionConfig::default(),
            coefficients: None,
            intercept_value: None,
            _state: PhantomData,
        }
    }

    pub fn intercept(mut self, intercept: bool) -> Self {
        self.config.intercept = intercept;
        self
    }

    pub fn penalty(mut self, penalty: Penalty) -> Self {
        self.config.penalty = penalty;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn solver(mut self, solver: Solver) -> Self {
        self.config.solver = solver;
        self
    }

    /// Set learning rate. Only effective for GradientDescent and Sgd solvers.
    /// For other solvers, this is a no-op.
    pub fn learning_rate(self, lr: f64) -> Self {
        match self.config.solver {
            Solver::GradientDescent { grad_tol, .. } => {
                self.solver(Solver::GradientDescent { learning_rate: lr, grad_tol })
            }
            Solver::Sgd { batch_size, .. } => {
                self.solver(Solver::Sgd { batch_size, learning_rate: lr })
            }
            _ => self,
        }
    }

    /// Set gradient tolerance. Only effective for GradientDescent solver.
    /// For other solvers, this is a no-op.
    pub fn grad_tol(self, grad_tol: f64) -> Self {
        match self.config.solver {
            Solver::GradientDescent { learning_rate, .. } => {
                self.solver(Solver::GradientDescent { learning_rate, grad_tol })
            }
            _ => self,
        }
    }

    /// Set batch size. Only effective for Sgd solver.
    /// For other solvers, this is a no-op.
    pub fn batch_size(self, batch_size: usize) -> Self {
        match self.config.solver {
            Solver::Sgd { learning_rate, .. } => {
                self.solver(Solver::Sgd { batch_size, learning_rate })
            }
            _ => self,
        }
    }

    /// Set memory size. Only effective for Lbfgs solver.
    /// For other solvers, this is a no-op.
    pub fn memory_size(self, memory_size: usize) -> Self {
        match self.config.solver {
            Solver::Lbfgs { .. } => {
                self.solver(Solver::Lbfgs { memory_size })
            }
            _ => self,
        }
    }
}

impl Component<Unfitted> for LogisticRegression<Unfitted> {
    type NextState = Fitted;
    type Output = LogisticRegression<Fitted>;
}

impl SupervisedEstimator<Unfitted> for LogisticRegression<Unfitted> {
    fn fit_supervised(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Output> {
        super::fit::fit(&self.config, x, y)
    }
}

impl LogisticRegression<Fitted> {
    pub(crate) fn new_fitted(
        config: LogisticRegressionConfig,
        coefficients: Vector,
        intercept_value: f64,
    ) -> Self {
        Self {
            config,
            coefficients: Some(coefficients),
            intercept_value: Some(intercept_value),
            _state: PhantomData,
        }
    }

    pub fn coefficients(&self) -> &Vector {
        self.coefficients.as_ref().unwrap()
    }

    pub fn intercept_value(&self) -> f64 {
        self.intercept_value.unwrap_or(0.0)
    }

    pub fn config(&self) -> &LogisticRegressionConfig {
        &self.config
    }

    pub fn predict_proba(&self, x: MatrixView<'_>) -> Result<Vector> {
        ops::ensure_nonempty_mat(x)?;
        let coef = self.coefficients.as_ref().unwrap();
        if x.ncols() != coef.len() {
            return Err(Error::invalid_shape(format!(
                "Input features mismatch: model has {}, input has {}",
                coef.len(),
                x.ncols()
            )));
        }
        let mut z = ops::gemv(x, coef.view())?;
        let intercept = self.intercept_value.unwrap_or(0.0);
        if intercept != 0.0 {
            z.mapv_inplace(|v| v + intercept);
        }
        z.mapv_inplace(|v| ops::sigmoid(v));
        Ok(z)
    }
}

impl Component<Fitted> for LogisticRegression<Fitted> {
    type NextState = Fitted;
    type Output = Vector;
}

impl Predictor for LogisticRegression<Fitted> {
    fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        let proba = self.predict_proba(x)?;
        let labels = proba.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 });
        Ok(labels)
    }
}
