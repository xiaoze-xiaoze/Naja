use crate::core::Result;
use crate::core::compute::types::{Matrix, MatrixView, Vector, VectorView};
use crate::core::traits::{
    Unfitted, Fitted,
    Predictor, Transformer, FittableTransformer,
    SupervisedEstimator, UnsupervisedEstimator,
    InversibleTransformer,
};
use std::marker::PhantomData;

pub struct Pipeline2<P, E, S = Unfitted> {
    pub preprocessor: P,
    pub estimator: E,
    _state: PhantomData<S>,
}

impl<P, E> Pipeline2<P, E, Unfitted> {
    pub fn new(preprocessor: P, estimator: E) -> Self {
        Self { preprocessor, estimator, _state: PhantomData }
    }
}

impl<P, E> Pipeline2<P, E, Unfitted>
where
    P: FittableTransformer,
    E: SupervisedEstimator<Unfitted>,
    P::Output: Transformer,
    E::Output: Predictor,
{
    pub fn fit_supervised(self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Pipeline2<P::Output, E::Output, Fitted>> {
        let (fitted_preprocessor, _x_transformed) = self.preprocessor.fit_transform(x)?;
        let x_transformed_for_y = fitted_preprocessor.transform(x)?;
        let fitted_estimator = self.estimator.fit_supervised(x_transformed_for_y.view(), y)?;
        Ok(Pipeline2 { preprocessor: fitted_preprocessor, estimator: fitted_estimator, _state: PhantomData })
    }
}

impl<P, E> Pipeline2<P, E, Unfitted>
where
    P: FittableTransformer,
    E: UnsupervisedEstimator<Unfitted>,
    P::Output: Transformer,
    E::Output: Predictor,
{
    pub fn fit_unsupervised(self, x: MatrixView<'_>) -> Result<Pipeline2<P::Output, E::Output, Fitted>> {
        let (fitted_preprocessor, _x_transformed) = self.preprocessor.fit_transform(x)?;
        let x_transformed_for_fit = fitted_preprocessor.transform(x)?;
        let fitted_estimator = self.estimator.fit_unsupervised(x_transformed_for_fit.view())?;
        Ok(Pipeline2 { preprocessor: fitted_preprocessor, estimator: fitted_estimator, _state: PhantomData })
    }
}

impl<P, M> Pipeline2<P, M, Fitted>
where
    P: Transformer,
    M: Predictor,
{
    pub fn predict(&self, x: MatrixView<'_>) -> Result<Vector> {
        let x_transformed = self.preprocessor.transform(x)?;
        self.estimator.predict(x_transformed.view())
    }
}

impl<P, M> Pipeline2<P, M, Fitted>
where
    P: InversibleTransformer,
    M: Predictor,
{
    pub fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix> {
        self.preprocessor.inverse_transform(x)
    }
}

pub fn pipeline<P, E>(preprocessor: P, estimator: E) -> Pipeline2<P, E, Unfitted> {
    Pipeline2::new(preprocessor, estimator)
}
