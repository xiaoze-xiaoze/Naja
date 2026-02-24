# 快速入门

本指南帮助你快速上手 Naja，包含完整的端到端示例。

## 环境准备

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

---

## 第一个模型：线性回归

### 1. 准备数据

```rust
use ndarray::{Array1, Array2};

let x: Array2<f64> = Array2::from_shape_vec(
    (5, 2),
    vec![
        1.0, 2.0,
        2.0, 3.0,
        3.0, 4.0,
        4.0, 5.0,
        5.0, 6.0,
    ],
).unwrap();

let y: Array1<f64> = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
```

### 2. 创建并训练模型

```rust
use naja::algorithms::linrg::LinearRegression;
use naja::core::traits::{SupervisedEstimator, Predictor};

let model = LinearRegression::new()
    .intercept(true);

let fitted = model.fit_supervised(x.view(), y.view())?;
```

### 3. 预测

```rust
let y_pred = fitted.predict(x.view())?;
println!("Predictions: {:?}", y_pred);
```

### 4. 评估

```rust
use naja::metrics;

let rmse = metrics::regressor::rmse(y.view(), y_pred.view())?;
let r2 = metrics::regressor::r2_score(y.view(), y_pred.view())?;

println!("RMSE: {:.4}", rmse);
println!("R²: {:.4}", r2);
```

---

## 使用 Pipeline

Pipeline 将预处理与模型训练封装为单一流程，避免数据泄露。

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::linrg::LinearRegression;
use naja::pipeline::pipeline;
use naja::core::traits::{SupervisedEstimator, Predictor};

// 构建流水线
let preprocessor = StandardScaler::new();
let estimator = LinearRegression::new().intercept(true);
let pipe = pipeline(preprocessor, estimator);

// 训练（自动完成：fit scaler -> transform -> fit model）
let fitted = pipe.fit_supervised(x_train.view(), y_train.view())?;

// 预测（自动完成：transform -> predict）
let y_pred = fitted.predict(x_test.view())?;
```

---

## 使用 Dataset 进行数据切分

```rust
use naja::core::data::Dataset;

let dataset = Dataset::new(x_train, y_train)?;
let (train, test) = dataset.split(0.2)?;

// train.records, train.targets
// test.records, test.targets
```

---

## 分类示例

```rust
use naja::algorithms::logrg::LogisticRegression;
use naja::core::traits::{SupervisedEstimator, Predictor, ProbabilisticPredictor};

let model = LogisticRegression::new()
    .intercept(true)
    .max_iter(100);

let fitted = model.fit_supervised(x.view(), y.view())?;

// 类别预测
let labels = fitted.predict(x_test.view())?;

// 概率预测
let probs = fitted.predict_proba(x_test.view())?;
```

---

## 下一步

- [核心 Trait](../core/traits.md) — 理解 typestate 模式
- [预处理](../preprocessing/index.md) — 选择合适的 Scaler
- [流水线](../pipeline/index.md) — 组合预处理与模型
- [指标](../metrics/index.md) — 评估模型性能
- [算法文档](../algorithms/index.md) — 查看支持的算法
