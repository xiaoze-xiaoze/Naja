# Core Traits（核心抽象）

`core` 是 Naja 的最小核心层，为算法实现提供基础抽象。所有算法模块依赖 core，但 core 不依赖任何算法模块。

## 模块组成

| 模块 | 职责 | 文档 |
|------|------|------|
| `traits` | 核心抽象 — fit/predict/transform trait | 本文档 |
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

---

## Traits 概述

核心 trait 定义了模型的训练、预测与变换接口，通过 typestate 模式在编译期区分拟合前后能力，保证算法统一范式与可组合性。

## 模块位置

- `src/core/traits.rs`

## 设计理念

采用 **Typestate 模式**：
- 使用 `Unfitted` / `Fitted` 状态标记类型参数
- `fit` 方法消费 `Unfitted` 状态，返回 `Fitted` 状态
- 编译期防止在未拟合对象上调用 `transform` / `predict`

```
Model<Unfitted> --fit()--> Model<Fitted>
                        |
                        +-- predict() / transform()
```

## 状态类型

| 类型 | 含义 |
|------|------|
| `State` | 状态标记 trait（sealed，外部无法实现） |
| `Unfitted` | 未拟合状态 |
| `Fitted` | 已拟合状态 |

## 核心 Trait

### 状态转换

| Trait | 关联类型 | 含义 |
|-------|----------|------|
| `Component<S>` | `NextState: State` | 定义状态转换目标 |
| | `Output` | 该状态下的输出类型 |

### 训练接口

| Trait | 签名 | 含义 | 约束 |
|-------|------|------|------|
| `SupervisedEstimator<S>` | `fn fit_supervised(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Output>` | 监督学习训练 | `Component<S, NextState=Fitted>` |
| `UnsupervisedEstimator<S>` | `fn fit_unsupervised(&self, x: MatrixView<'_>) -> Result<Self::Output>` | 无监督学习训练 | `Component<S, NextState=Fitted>` |
| `FittableTransformer` | `fn fit(self, x: MatrixView<'_>) -> Result<Self::Output>` | 可拟合变换器 | `Component<Unfitted, NextState=Fitted>` |
| | `fn fit_transform(self, x: MatrixView<'_>) -> Result<(Self::Output, Matrix)>` | 拟合并变换（默认实现） | |
| `PartialFit` | `fn partial_fit(&mut self, x: MatrixView<'_>, y: Option<VectorView<'_>>) -> Result<()>` | 增量/在线学习 | - |

### 推理接口

| Trait | 签名 | 含义 | 约束 |
|-------|------|------|------|
| `Predictor` | `fn predict(&self, x: MatrixView<'_>) -> Result<Vector>` | 预测输出 | `Component<Fitted, Output=Vector>` |
| `ProbabilisticPredictor` | `fn predict_proba(&self, x: MatrixView<'_>) -> Result<Matrix>` | 概率分布预测 | `Component<Fitted, Output=Matrix>` |
| `Transformer` | `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 数据变换 | `Component<Fitted, Output=Matrix>` |
| `InversibleTransformer` | `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 逆变换 | 继承 `Transformer` |

## Trait 层次结构

```
State (sealed)
  ├── Unfitted
  └── Fitted

Component<S: State>
  ├── SupervisedEstimator<S>   --fit_supervised-->  Predictor
  ├── UnsupervisedEstimator<S> --fit_unsupervised--> Predictor
  └── FittableTransformer      --fit-->  Transformer

Transformer
  └── InversibleTransformer (+ inverse_transform)
```

## 范式流程

| 阶段 | 接口 | 输入 | 输出 |
|------|------|------|------|
| Define | 构造函数 | 超参数 | `Model<Unfitted>` |
| Fit | `SupervisedEstimator` / `UnsupervisedEstimator` / `FittableTransformer` | 数据 | `Model<Fitted>` 或 `Solution` |
| Predict | `Predictor` / `ProbabilisticPredictor` | 新数据 | 预测结果 |
| Transform | `Transformer` / `InversibleTransformer` | 数据 | 变换结果 |

## 使用示例

### 监督学习

```rust
use naja::algorithms::linrg::LinearRegression;
use naja::core::traits::{SupervisedEstimator, Predictor};
let model = LinearRegression::new().intercept(true);
let fitted = model.fit_supervised(x_train.view(), y_train.view())?;
let y_pred = fitted.predict(x_test.view())?;
```

### 无监督学习

```rust
use naja::algorithms::kmeans::KMeans;
use naja::core::traits::{UnsupervisedEstimator, Predictor};
let model = KMeans::new().k(3);
let fitted = model.fit_unsupervised(x.view())?;
let labels = fitted.predict(x.view())?;
```

### 预处理变换器

```rust
use naja::preprocessing::StandardScaler;
use naja::core::traits::{FittableTransformer, Transformer, InversibleTransformer};
let scaler = StandardScaler::new();
let fitted = scaler.fit(x_train.view())?;
let x_scaled = fitted.transform(x_test.view())?;
let x_original = fitted.inverse_transform(x_scaled.view())?;
```

### 增量学习

```rust
use naja::preprocessing::StandardScaler;
use naja::core::traits::{FittableTransformer, PartialFit};
let mut fitted = StandardScaler::new().fit(batch1.view())?;
for batch in data_stream {
    fitted.partial_fit(batch.view(), None)?;
}
```

## 注意事项

- `State` trait 使用 sealed pattern，外部 crate 无法实现自己的状态类型
- `PartialFit` 的 `y` 参数为 `Option`，无监督场景传 `None`
- `FittableTransformer::fit` 消费 `self`，不可重复调用
- 所有 trait 方法返回 `core::Result<T>`，统一错误处理
