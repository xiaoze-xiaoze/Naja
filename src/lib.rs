pub mod algorithms;
pub mod core;
pub mod io;
pub mod metrics;
pub mod pipeline;
pub mod preprocessing;

pub use core::{Error, Result};

pub mod prelude {
    pub use crate::core::traits::{
        Component, State, Unfitted, Fitted,
        Predictor, ProbabilisticPredictor,
        Transformer, InversibleTransformer, FittableTransformer,
        SupervisedEstimator, UnsupervisedEstimator,
        PartialFit,
    };
    pub use crate::pipeline::Pipeline2;
    pub use crate::core::Result;
    pub use crate::preprocessing::{StandardScaler, MinMaxScaler, RobustScaler};
    pub use crate::algorithms::linrg::LinearRegression;
}
