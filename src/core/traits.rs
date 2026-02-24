mod sealed { pub trait Sealed {} }

pub trait State: sealed::Sealed + Clone + Copy + Send + Sync {}

#[derive(Debug, Clone, Copy, Default)]
pub struct Unfitted;

#[derive(Debug, Clone, Copy, Default)]
pub struct Fitted;

impl sealed::Sealed for Unfitted {}
impl sealed::Sealed for Fitted {}
impl State for Unfitted {}
impl State for Fitted {}

use crate::core::Result;
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};

pub trait Component<S: State> {
    type NextState: State;
    type Output;
}

pub trait SupervisedEstimator<S: State>: Component<S, NextState = Fitted>
where
    Self::Output: Predictor,
{
    fn fit_supervised(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Output>;
}

pub trait UnsupervisedEstimator<S: State>: Component<S, NextState = Fitted>
where
    Self::Output: Predictor,
{
    fn fit_unsupervised(&self, x: MatrixView<'_>) -> Result<Self::Output>;
}

pub trait Predictor: Component<Fitted, Output = Vector> {
    fn predict(&self, x: MatrixView<'_>) -> Result<Vector>;
}

pub trait ProbabilisticPredictor: Component<Fitted, Output = Matrix> {
    fn predict_proba(&self, x: MatrixView<'_>) -> Result<Matrix>;
}

pub trait Transformer: Component<Fitted, Output = Matrix> {
    fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>;
}

pub trait InversibleTransformer: Transformer {
    fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>;
}

pub trait FittableTransformer: Component<Unfitted, NextState = Fitted> + Sized
where
    Self::Output: Transformer,
{
    fn fit(self, x: MatrixView<'_>) -> Result<Self::Output>;

    fn fit_transform(self, x: MatrixView<'_>) -> Result<(Self::Output, Matrix)> {
        let fitted = self.fit(x)?;
        let transformed = fitted.transform(x)?;
        Ok((fitted, transformed))
    }
}

pub trait PartialFit {
    fn partial_fit(&mut self, x: MatrixView<'_>, y: Option<VectorView<'_>>) -> Result<()>;
}
