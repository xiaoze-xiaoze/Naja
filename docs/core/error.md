# error

## 概述

`core::error` 统一库内错误类型与 `Result` 别名，保证跨模块的错误语义一致。所有公共 API 返回 `core::Result<T>`，便于错误传播与处理。

## 模块位置

- `src/core/error.rs`

## 类型定义

### Result

```rust
pub type Result<T> = core::result::Result<T, Error>;
```

统一返回类型，所有核心模块的标准返回值。

### Error

```rust
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
```

| 变体 | 含义 | 典型场景 |
|------|------|----------|
| `LinAlg` | 线性代数计算失败 | 矩阵奇异、Cholesky 分解失败 |
| `InvalidShape` | 形状/维度不匹配 | 矩阵乘法维度不兼容、向量长度不一致 |
| `InvalidParam` | 参数非法 | 负数 K 值、无效正则化系数 |
| `EmptyInput` | 输入为空 | 空矩阵、空向量 |
| `InvalidState` | 非法状态 | 在未拟合对象上调用 transform |
| `BackendUnavailable` | 后端不可用 | 未启用 `faer-backend` 调用线性求解器 |

---

## 构造方法

| 方法 | 签名 | 含义 |
|------|------|------|
| `lin_alg` | `fn lin_alg(msg: impl Into<String>) -> Self` | 创建线性代数错误 |
| `invalid_shape` | `fn invalid_shape(msg: impl Into<String>) -> Self` | 创建形状不匹配错误 |
| `invalid_param` | `fn invalid_param(name: impl Into<String>, msg: impl Into<String>) -> Self` | 创建参数非法错误 |
| `empty_input` | `fn empty_input(name: impl Into<String>) -> Self` | 创建空输入错误 |
| `invalid_state` | `fn invalid_state(msg: impl Into<String>) -> Self` | 创建非法状态错误 |
| `backend_unavailable` | `fn backend_unavailable(name: impl Into<String>, msg: impl Into<String>) -> Self` | 创建后端不可用错误 |

---

## 使用示例

### 错误传播

```rust
use naja::core::{Error, Result};

fn my_function(x: MatrixView<'_>) -> Result<Vector> {
    ops::ensure_nonempty_mat(x)?;
    
    if x.ncols() == 0 {
        return Err(Error::invalid_shape("matrix has zero columns"));
    }
    
    // ...
    Ok(result)
}
```

### 错误匹配

```rust
use naja::core::Error;

match model.fit(&x, &y) {
    Ok(fitted) => { /* ... */ },
    Err(Error::InvalidShape(msg)) => eprintln!("Shape error: {}", msg),
    Err(Error::LinAlg(msg)) => eprintln!("Linear algebra error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### 与 `?` 运算符配合

```rust
use naja::core::Result;

fn train_pipeline(x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector> {
    let scaler = StandardScaler::new().fit(x)?;
    let x_scaled = scaler.transform(x)?;
    
    let model = LinearRegression::new();
    let fitted = model.fit_supervised(x_scaled.view(), y)?;
    
    fitted.predict(x_scaled.view())
}
```

---

## 注意事项

- `Error` 实现 `Clone`，可安全地在多处使用同一错误
- 使用 `thiserror` crate 派生 `std::error::Error`，可与 anyhow 等库兼容
- 错误消息设计为可读的英文描述，便于调试
