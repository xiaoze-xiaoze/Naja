use std::marker::PhantomData;
use crate::core::{Result, Error};
use crate::core::compute::types::{MatrixView, Vector, VectorView};
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
    ClosedForm,
    GradientDescent { learning_rate: f64, grad_tol: f64 },
    CoordinateDescent,
}

impl Default for Solver {
    fn default() -> Self { Solver::ClosedForm }
}

#[derive(Debug, Clone)]
pub struct LinearRegressionConfig {
    pub intercept: bool,
    pub penalty: Penalty,
    pub max_iter: usize,
    pub tol: f64,
    pub solver: Solver,
}

impl Default for LinearRegressionConfig {
    fn default() -> Self {
        Self { intercept: true, penalty: Penalty::None, max_iter: 1000, tol: 1e-4, solver: Solver::ClosedForm }
    }
}

#[derive(Debug, Clone)]
pub struct LinearRegression<S: State = Unfitted> {
    config: LinearRegressionConfig,
    coefficients: Option<Vector>,
    intercept_value: Option<f64>,
    _state: PhantomData<S>,
}

impl Default for LinearRegression<Unfitted> {
    fn default() -> Self { Self::new() }
}

impl LinearRegression<Unfitted> {
    pub fn new() -> Self {
        Self {
            config: LinearRegressionConfig::default(),
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
}

impl Component<Unfitted> for LinearRegression<Unfitted> {
    type NextState = Fitted;
    type Output = LinearRegression<Fitted>;
}

impl SupervisedEstimator<Unfitted> for LinearRegression<Unfitted> {
    fn fit_supervised(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Output> {
        super::fit::fit(&self.config, x, y)
    }
}

impl LinearRegression<Fitted> {
    pub(crate) fn new_fitted(
        config: LinearRegressionConfig,
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

    pub fn config(&self) -> &LinearRegressionConfig {
        &self.config
    }
}

impl Component<Fitted> for LinearRegression<Fitted> {
    type NextState = Fitted;
    type Output = Vector;
}

impl Predictor for LinearRegression<Fitted> {
    fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        ops::ensure_nonempty_mat(x)?;
        let coef = self.coefficients.as_ref().unwrap();
        if x.ncols() != coef.len() {
            return Err(Error::invalid_shape(format!(
                "Input features mismatch: model has {}, input has {}",
                coef.len(),
                x.ncols()
            )));
        }
        let mut y_pred = ops::gemv(x, coef.view())?;
        let intercept = self.intercept_value.unwrap_or(0.0);
        if intercept != 0.0 {
            y_pred.mapv_inplace(|v| v + intercept);
        }
        Ok(y_pred)
    }
}
