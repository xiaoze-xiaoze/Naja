# core 模块

## 概述

`core` 是 Naja 的最小核心层，为算法实现提供基础抽象。所有算法模块依赖 core，但 core 不依赖任何算法模块。

## 设计理念

- **最小化**：仅包含算法所需的公共抽象
- **Typestate 模式**：编译期区分拟合前后状态，防止运行时错误
- **统一错误处理**：所有模块使用相同的 `Result<T>` 类型

## 组成部分

| 模块 | 职责 | 文档 |
|------|------|------|
| `traits` | 核心抽象 — fit/predict/transform trait | [traits.md](traits.md) |
| `error` | 错误处理 — 统一 Error 与 Result | [error.md](error.md) |
| `compute` | 数值运算 — ndarray/faer 封装 | [compute.md](compute.md) |
| `data` | 数据容器 — Dataset 与校验 | [data.md](data.md) |

## 模块依赖关系

```
core/
├── error.rs          # 无依赖
├── traits.rs         # 依赖 error, compute::types
├── compute/          # 依赖 error
│   ├── types.rs
│   └── ops.rs
└── data/             # 依赖 error, compute
    ├── dataset.rs
    └── validate.rs
```

## 核心 Trait 概览

```rust
// 状态类型
pub struct Unfitted;
pub struct Fitted;

// 训练接口
pub trait SupervisedEstimator<S: State> { ... }
pub trait UnsupervisedEstimator<S: State> { ... }
pub trait FittableTransformer { ... }

// 推理接口
pub trait Predictor { ... }
pub trait Transformer { ... }
pub trait InversibleTransformer { ... }
```

详细说明请参阅 [traits.md](traits.md)。

## 使用示例

```rust
use naja::core::Result;
use naja::core::compute::types::{Matrix, Vector, MatrixView, VectorView};
use naja::core::traits::{SupervisedEstimator, Predictor};

fn train_and_predict<M>(model: M, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector>
where
    M: SupervisedEstimator<Unfitted>,
    M::Output: Predictor,
{
    let fitted = model.fit_supervised(x, y)?;
    fitted.predict(x)
}
```
