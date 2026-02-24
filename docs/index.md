# Naja 文档

## 概述

Naja 是一个面向工程使用的 Rust 机器学习库，提供一致的 Model / Fit / Predict-Transform 使用范式。

---

## 快速入门

### 环境准备

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
naja = { path = "..." }  # 或 crates.io 版本
ndarray = "0.15"
```

如需使用高性能线性求解器，启用 `faer-backend` feature：

```toml
[dependencies]
naja = { path = "...", features = ["faer-backend"] }
```

### 第一个模型：线性回归

```rust
use ndarray::{Array1, Array2};
use naja::algorithms::linrg::LinearRegression;
use naja::core::traits::{SupervisedEstimator, Predictor};
use naja::metrics;

// 准备数据
let x: Array2<f64> = Array2::from_shape_vec(
    (5, 2),
    vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
).unwrap();
let y: Array1<f64> = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

// 创建并训练模型
let model = LinearRegression::new().intercept(true);
let fitted = model.fit_supervised(x.view(), y.view())?;

// 预测
let y_pred = fitted.predict(x.view())?;

// 评估
let rmse = metrics::regressor::rmse(y.view(), y_pred.view())?;
let r2 = metrics::regressor::r2_score(y.view(), y_pred.view())?;
```

### 使用 Pipeline

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::linrg::LinearRegression;
use naja::pipeline::pipeline;
use naja::core::traits::{SupervisedEstimator, Predictor};

let preprocessor = StandardScaler::new();
let estimator = LinearRegression::new().intercept(true);
let pipe = pipeline(preprocessor, estimator);

let fitted = pipe.fit_supervised(x_train.view(), y_train.view())?;
let y_pred = fitted.predict(x_test.view())?;
```

---

## API 设计范式

| 阶段 | 方法 | 职责 |
|:----:|:----:|:----:|
| Model | `Model::new().param(...)` | 创建模型并链式配置超参数 |
| Fit | `fit(&x, &y)` / `fit(&x)` | 训练/求解 |
| Predict / Transform | `predict(&x)` / `transform(&x)` | 推理/变换 |

---

## 模块文档

### 核心模块 (core)

| 模块 | 说明 |
|------|------|
| [traits](core/traits.md) | 核心抽象 — fit/predict/transform trait |
| [error](core/error.md) | 错误处理 — 统一 Result 与 Error 类型 |
| [compute](core/compute.md) | 数值运算 — ndarray/faer 封装 |
| [data](core/data.md) | 数据容器 — Dataset 与校验函数 |

### 预处理与流水线

| 模块 | 说明 |
|------|------|
| [preprocessing](preprocessing/index.md) | 数据预处理 — StandardScaler / MinMaxScaler / RobustScaler |
| [pipeline](pipeline/index.md) | 流水线 — 预处理与模型的组合 |

### 评估与导出

| 模块 | 说明 |
|------|------|
| [metrics](metrics/index.md) | 指标库 — 回归/分类/聚类评估指标 |
| [io](io/index.md) | 模型导出 — ONNX（规划中） |
