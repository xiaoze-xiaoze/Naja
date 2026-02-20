use thiserror::Error;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Linear Algebra Error: {0}")]
    LinAlg(String),
    #[error("Invalid Shape: {0}")]
    InvalidShape(String),
    #[error("Invalid Parameter '{name}': {msg}")]
    InvalidParam { name: String, msg: String },
    #[error("Empty Input: {0}")]
    EmptyInput(String),
    #[error("Invalid State: {0}")]
    InvalidState(String),
    #[error("Backend Unavailable: {name} - {msg}")]
    BackendUnavailable { name: String, msg: String },
}

impl Error {
    pub fn lin_alg(msg: impl Into<String>) -> Self { Error::LinAlg(msg.into()) }
    pub fn invalid_shape(msg: impl Into<String>) -> Self { Error::InvalidShape(msg.into()) }
    pub fn invalid_param(name: impl Into<String>, msg: impl Into<String>) -> Self {  Error::InvalidParam { name: name.into(), msg: msg.into() } }
    pub fn empty_input(name: impl Into<String>) -> Self { Error::EmptyInput(name.into()) }
    pub fn invalid_state(msg: impl Into<String>) -> Self { Error::InvalidState(msg.into()) }
    pub fn backend_unavailable(name: impl Into<String>, msg: impl Into<String>) -> Self { Error::BackendUnavailable { name: name.into(), msg: msg.into() } }
}