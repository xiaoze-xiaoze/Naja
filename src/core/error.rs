use core::fmt;

pub type Result<T> = core::result::Result<T, Error>;

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    EmptyInput { name: String},                            // 输入为空
    InvalidShape { msg: String},                           // 形状不匹配
    InvalidParam { name: String, msg: String},             // 无效参数
    NonFinite { name: String},                             // 非有限值
    NotConverged { name: String},                          // 未收敛
    LinAlg { msg: String},                                 // 线性代数错误
    BackendUnavailable { feature: String, msg: String},    // 后端不可用
    Internal { msg: String}                                // 内部错误
}

impl Error {
    pub fn empty_input(name: impl Into<String>) -> Self { Self::EmptyInput { name: name.into() } }
    pub fn invalid_shape(msg: impl Into<String>) -> Self { Self::InvalidShape { msg: msg.into() } }
    pub fn invalid_param(name: impl Into<String>, msg: impl Into<String>) -> Self { Self::InvalidParam { name: name.into(), msg: msg.into() } }
    pub fn non_finite(name: impl Into<String>) -> Self { Self::NonFinite { name: name.into() } }
    pub fn not_converged(name: impl Into<String>) -> Self { Self::NotConverged { name: name.into() } }
    pub fn lin_alg(msg: impl Into<String>) -> Self { Self::LinAlg { msg: msg.into() } }
    pub fn backend_unavailable(feature: impl Into<String>, msg: impl Into<String>) -> Self { Self::BackendUnavailable { feature: feature.into(), msg: msg.into() } }
    pub fn internal(msg: impl Into<String>) -> Self { Self::Internal { msg: msg.into() } }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { name } => write!(f, "Empty input: {}", name),
            Self::InvalidShape { msg } => write!(f, "Invalid shape: {}", msg),
            Self::InvalidParam { name, msg } => write!(f, "Invalid parameter {}: {}", name, msg),
            Self::NonFinite { name } => write!(f, "Non-finite value: {}", name),
            Self::NotConverged { name } => write!(f, "Not converged: {}", name),
            Self::LinAlg { msg } => write!(f, "Linear algebra error: {}", msg),
            Self::BackendUnavailable { feature, msg } => write!(f, "Backend unavailable for {}: {}", feature, msg),
            Self::Internal { msg } => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}