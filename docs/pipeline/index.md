# pipeline（流水线）

## 概述

`pipeline` 模块提供预处理与估计器的组合工具，将多个处理步骤封装为单一对象，支持统一的 `fit` / `predict` 调用。

## 模块位置

- `src/pipeline/mod.rs`

## 设计理念

- **封装复杂性**：将预处理 + 模型训练封装为单一流程
- **防止数据泄露**：确保预处理参数仅从训练集学习
- **类型安全**：编译期检查流水线各阶段的兼容性

## 类型定义

### Pipeline2<P, E, S>

```rust
pub struct Pipeline2<P, E, S = Unfitted> {
    pub preprocessor: P,
    pub estimator: E,
    _state: PhantomData<S>,
}
```

| 类型参数 | 含义 |
|----------|------|
| `P` | 预处理器类型（需实现 `FittableTransformer`） |
| `E` | 估计器类型（需实现 `SupervisedEstimator` 或 `UnsupervisedEstimator`） |
| `S` | 状态（`Unfitted` 或 `Fitted`） |

### 构造函数

```rust
pub fn pipeline<P, E>(preprocessor: P, estimator: E) -> Pipeline2<P, E, Unfitted>
```

## 方法

### Pipeline2<P, E, Unfitted>

| 方法 | 签名 | 含义 | 约束 |
|------|------|------|------|
| `new` | `fn new(preprocessor: P, estimator: E) -> Self` | 创建未拟合流水线 | - |
| `fit_supervised` | `fn fit_supervised(self, x, y) -> Result<Pipeline2<P::Output, E::Output, Fitted>>` | 监督学习训练 | `P: FittableTransformer`, `E: SupervisedEstimator<Unfitted>` |
| `fit_unsupervised` | `fn fit_unsupervised(self, x) -> Result<Pipeline2<P::Output, E::Output, Fitted>>` | 无监督学习训练 | `P: FittableTransformer`, `E: UnsupervisedEstimator<Unfitted>` |

### Pipeline2<P, E, Fitted>

| 方法 | 签名 | 含义 | 约束 |
|------|------|------|------|
| `predict` | `fn predict(&self, x: MatrixView<'_>) -> Result<Vector>` | 预测 | `P: Transformer`, `E: Predictor` |
| `inverse_transform` | `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 预处理逆变换 | `P: InversibleTransformer`, `E: Predictor` |

## 使用示例

### 监督学习流水线

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::linrg::LinearRegression;
use naja::pipeline::pipeline;
use naja::core::traits::{SupervisedEstimator, Predictor};

let preprocessor = StandardScaler::new();
let estimator = LinearRegression::new().intercept(true);
let pipe = pipeline(preprocessor, estimator);

// 训练：自动完成 fit_transform(preprocessor) + fit_supervised(estimator)
let fitted = pipe.fit_supervised(x_train.view(), y_train.view())?;

// 预测：自动完成 transform(preprocessor) + predict(estimator)
let y_pred = fitted.predict(x_test.view())?;
```

### 无监督学习流水线

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::kmeans::KMeans;
use naja::pipeline::pipeline;
use naja::core::traits::{UnsupervisedEstimator, Predictor};

let preprocessor = StandardScaler::new();
let estimator = KMeans::new().k(3);
let pipe = pipeline(preprocessor, estimator);

let fitted = pipe.fit_unsupervised(x.view())?;
let labels = fitted.predict(x.view())?;
```

### 逆变换

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::linrg::LinearRegression;
use naja::pipeline::pipeline;

let preprocessor = StandardScaler::new();
let estimator = LinearRegression::new();
let pipe = pipeline(preprocessor, estimator);
let fitted = pipe.fit_supervised(x_train.view(), y_train.view())?;

// 仅对预处理部分做逆变换
let x_original = fitted.inverse_transform(x_scaled.view())?;
```

## 类型约束说明

流水线的类型约束确保各阶段兼容：

```
Pipeline2<Preprocessor, Estimator, Unfitted>
    |
    +-- fit_supervised: requires
    |       Preprocessor: FittableTransformer (fit -> Fitted, Output: Transformer)
    |       Estimator: SupervisedEstimator<Unfitted> (fit_supervised -> Fitted, Output: Predictor)
    |
    v
Pipeline2<Preprocessor::Output, Estimator::Output, Fitted>
    |
    +-- predict: requires
            Preprocessor::Output: Transformer
            Estimator::Output: Predictor
```

## 注意事项

1. **数据泄露防护**：预处理参数（如 mean/std）仅从训练集学习，测试集通过同一 fitted preprocessor 变换
2. **所有权转移**：`fit_supervised` / `fit_unsupervised` 消费 `self`，返回新的 Fitted 流水线
3. **逆变换范围**：`inverse_transform` 仅对预处理部分有效，不涉及估计器
4. **目前仅支持单阶段预处理**：如需多阶段，可嵌套或后续扩展 `Pipeline3` 等

## 与直接调用对比

**无 Pipeline：**

```rust
let scaler = StandardScaler::new().fit(x_train.view())?;
let x_train_scaled = scaler.transform(x_train.view())?;
let model = LinearRegression::new();
let fitted = model.fit_supervised(x_train_scaled.view(), y_train.view())?;
let x_test_scaled = scaler.transform(x_test.view())?;
let y_pred = fitted.predict(x_test_scaled.view())?;
```

**使用 Pipeline：**

```rust
let pipe = pipeline(StandardScaler::new(), LinearRegression::new());
let fitted = pipe.fit_supervised(x_train.view(), y_train.view())?;
let y_pred = fitted.predict(x_test.view())?;
```
