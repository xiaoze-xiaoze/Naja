pub mod compute;
pub mod data;
pub mod error;
pub mod traits;

pub use error::{Error, Result};
pub use traits::{
    State, Unfitted, Fitted,
    Component,
    SupervisedEstimator, UnsupervisedEstimator,
    Predictor, ProbabilisticPredictor,
    Transformer, InversibleTransformer, FittableTransformer,
    PartialFit,
};
