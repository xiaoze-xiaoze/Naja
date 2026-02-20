use crate::core::Result;
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};

pub trait Predictor { fn predict(&self, x: MatrixView<'_>) -> Result<Vector>; }

pub trait ProbabilisticPredictor { fn predict_proba(&self, x: MatrixView<'_>) -> Result<Matrix>; }

pub trait Transformer { fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>; }

pub trait InverseTransformer { fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>; }

pub trait FitSupervised {
    type Object;
    fn fit(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Object>;
}

pub trait FitUnsupervised {
    type Object;
    fn fit(&self, x: MatrixView<'_>) -> Result<Self::Object>;
}

pub trait PartialFit { fn partial_fit(&mut self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<()>; }

pub trait SupervisedModel {}
pub trait UnsupervisedModel {}
