# preprocessing（数据预处理）

## 概述

`preprocessing` 提供特征缩放工具，统一采用 typestate 模式，支持 `fit` / `transform` / `inverse_transform` 三阶段范式。所有 Scaler 实现 `FittableTransformer`、`Transformer`、`InversibleTransformer` trait。

## 模块位置

- `src/preprocessing/mod.rs`
- `src/preprocessing/scaler/standard.rs`
- `src/preprocessing/scaler/minmax.rs`
- `src/preprocessing/scaler/robust.rs`

## Scaler 总览

| Scaler | 核心思想 | 适用场景 | 增量学习 |
|--------|----------|----------|----------|
| `StandardScaler` | 均值 0、标准差 1 | 数据近似正态分布 | ✓ |
| `MinMaxScaler` | 缩放到指定范围 | 需要有界输出（如神经网络） | ✓ |
| `RobustScaler` | 中位数 + IQR | 数据存在异常值 | ✗ |

---

## StandardScaler

标准化缩放器，将特征转换为均值 0、标准差 1 的分布。

### 公式

- **transform**: $x' = \frac{x - \mu}{\sigma}$
- **inverse_transform**: $x = x' \cdot \sigma + \mu$

### 类型定义

```rust
pub struct StandardScaler<S: State = Unfitted> {
    mean: Option<Vector>,
    std: Option<Vector>,
    // ...
}
```

### 方法

#### StandardScaler\<Unfitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `new` | `fn new() -> Self` | 创建未拟合实例 |
| `fit` | `fn fit(self, x: MatrixView<'_>) -> Result<StandardScaler<Fitted>>` | 计算均值和标准差 |

#### StandardScaler\<Fitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `transform` | `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 标准化变换 |
| `inverse_transform` | `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 逆变换恢复原尺度 |
| `partial_fit` | `fn partial_fit(&mut self, x: MatrixView<'_>, y: Option<VectorView<'_>>) -> Result<()>` | 增量更新统计量 |

### 使用示例

```rust
use naja::preprocessing::StandardScaler;
use naja::core::traits::{FittableTransformer, Transformer, InversibleTransformer};

let scaler = StandardScaler::new();
let fitted = scaler.fit(x_train.view())?;

let x_train_scaled = fitted.transform(x_train.view())?;
let x_test_scaled = fitted.transform(x_test.view())?;

// 恢复原尺度
let x_original = fitted.inverse_transform(x_train_scaled.view())?;

// 增量学习
let mut fitted = StandardScaler::new().fit(batch1.view())?;
for batch in stream {
    fitted.partial_fit(batch.view(), None)?;
}
```

### 注意事项

- 当某列 `std < 1e-8` 时，自动替换为 `1.0` 避免除零
- `partial_fit` 使用 Welford 在线算法更新均值和方差

---

## MinMaxScaler

归一化缩放器，将特征缩放到指定范围（默认 [0, 1]）。

### 公式

- **transform**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (range_{max} - range_{min}) + range_{min}$
- **inverse_transform**: $x = \frac{x' - range_{min}}{range_{max} - range_{min}} \cdot (x_{max} - x_{min}) + x_{min}$

### 类型定义

```rust
pub struct MinMaxScaler<S: State = Unfitted> {
    min: Option<Vector>,
    scale: Option<Vector>,
    feature_range: (f64, f64),
    // ...
}
```

### 参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `feature_range` | `(f64, f64)` | `(0.0, 1.0)` | 目标输出范围 |

### 方法

#### MinMaxScaler\<Unfitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `new` | `fn new() -> Self` | 创建未拟合实例 |
| `with_feature_range` | `fn with_feature_range(self, min: f64, max: f64) -> Self` | 设置目标范围 |
| `fit` | `fn fit(self, x: MatrixView<'_>) -> Result<MinMaxScaler<Fitted>>` | 计算每列 min/max |

#### MinMaxScaler\<Fitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `transform` | `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 归一化变换 |
| `inverse_transform` | `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 逆变换恢复原尺度 |
| `partial_fit` | `fn partial_fit(&mut self, x: MatrixView<'_>, y: Option<VectorView<'_>>) -> Result<()>` | 增量更新 min/max |

### 使用示例

```rust
use naja::preprocessing::MinMaxScaler;
use naja::core::traits::{FittableTransformer, Transformer, InversibleTransformer};

let scaler = MinMaxScaler::new()
    .with_feature_range(-1.0, 1.0);
let fitted = scaler.fit(x_train.view())?;

let x_scaled = fitted.transform(x_test.view())?;
// 输出范围: [-1.0, 1.0]
```

### 注意事项

- 当某列 `scale (max-min) < 1e-8` 时，自动替换为 `1.0` 避免除零
- `partial_fit` 会扩展 min/max 范围，但不会收缩

---

## RobustScaler

鲁棒缩放器，使用中位数和四分位距（IQR），对异常值不敏感。

### 公式

- **transform**: $x' = \frac{x - \text{median}}{\text{IQR}}$
- **inverse_transform**: $x = x' \cdot \text{IQR} + \text{median}$

其中 $\text{IQR} = Q_3 - Q_1$（默认第 75 百分位 - 第 25 百分位）

### 类型定义

```rust
pub struct RobustScaler<S: State = Unfitted> {
    median: Option<Vector>,
    iqr: Option<Vector>,
    quantile_range: (f64, f64),
    center: bool,
    scale: bool,
    // ...
}
```

### 参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `quantile_range` | `(f64, f64)` | `(0.25, 0.75)` | 计算 IQR 的分位数范围 |
| `center` | `bool` | `true` | 是否减去中位数 |
| `scale` | `bool` | `true` | 是否除以 IQR |

### 方法

#### RobustScaler\<Unfitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `new` | `fn new() -> Self` | 创建未拟合实例 |
| `with_quantile_range` | `fn with_quantile_range(self, q1: f64, q3: f64) -> Self` | 设置分位数范围 |
| `with_center` | `fn with_center(self, center: bool) -> Self` | 是否中心化 |
| `with_scale` | `fn with_scale(self, scale: bool) -> Self` | 是否缩放 |
| `fit` | `fn fit(self, x: MatrixView<'_>) -> Result<RobustScaler<Fitted>>` | 计算中位数和 IQR |

#### RobustScaler\<Fitted\>

| 方法 | 签名 | 含义 |
|------|------|------|
| `transform` | `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 鲁棒变换 |
| `inverse_transform` | `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>` | 逆变换恢复原尺度 |

### 使用示例

```rust
use naja::preprocessing::RobustScaler;
use naja::core::traits::{FittableTransformer, Transformer, InversibleTransformer};

let scaler = RobustScaler::new()
    .with_quantile_range(0.1, 0.9)
    .with_center(true)
    .with_scale(true);
let fitted = scaler.fit(x_train.view())?;

let x_scaled = fitted.transform(x_test.view())?;
```

### 注意事项

- 当某列 `IQR < 1e-8` 时，自动替换为 `1.0` 避免除零
- 不支持 `partial_fit`（分位数计算难以增量更新）
- `center=false` 且 `scale=false` 时，`transform` 返回原数据

---

## 通用 Trait 实现

所有 Scaler 均实现以下 trait：

| Trait | 方法 | 说明 |
|-------|------|------|
| `FittableTransformer` | `fit`, `fit_transform` | 支持 fit → Fitted 转换 |
| `Transformer` | `transform` | 数据变换 |
| `InversibleTransformer` | `inverse_transform` | 逆变换 |

部分 Scaler 额外实现：

| Trait | Scaler | 说明 |
|-------|--------|------|
| `PartialFit` | StandardScaler, MinMaxScaler | 增量学习 |

---

## 使用建议

1. **只用训练集 fit**：避免数据泄露，测试集只用同一 fitted 对象 transform
2. **维度一致性**：`transform` 输入的列数必须与 `fit` 时一致
3. **选择 Scaler**：
   - 数据近似正态分布 → StandardScaler
   - 需要固定范围输出 → MinMaxScaler
   - 数据存在异常值 → RobustScaler

---

## 与 Pipeline 组合

```rust
use naja::preprocessing::StandardScaler;
use naja::algorithms::linrg::LinearRegression;
use naja::pipeline::pipeline;
use naja::core::traits::SupervisedEstimator;

let preprocessor = StandardScaler::new();
let estimator = LinearRegression::new();
let pipe = pipeline(preprocessor, estimator);

let fitted_pipe = pipe.fit_supervised(x_train.view(), y_train.view())?;
let y_pred = fitted_pipe.predict(x_test.view())?;
```
