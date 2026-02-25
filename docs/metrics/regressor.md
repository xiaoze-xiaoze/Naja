# Regressor Metrics（回归指标）

回归评估指标，所有函数统一返回 `core::Result<f64>` 并进行输入校验。

## 模块位置

- `src/metrics/regressor.rs`

## 指标列表

| 函数 | 签名 | 公式 | 含义 |
|------|------|------|------|
| `mse` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差 |
| `rmse` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\sqrt{\text{MSE}}$ | 均方根误差 |
| `mae` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 平均绝对误差 |
| `r2_score` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | 决定系数 |

## 使用建议

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| MSE | 对大误差敏感 | 需要惩罚大偏差 |
| RMSE | 与原数据同量纲 | 直观解释误差大小 |
| MAE | 对异常值不敏感 | 数据含异常值 |
| R² | 无量纲，可比较 | 模型对比、解释方差比例 |

## 使用示例

```rust
use naja::metrics;
let mse = metrics::regressor::mse(y_true.view(), y_pred.view())?;
let rmse = metrics::regressor::rmse(y_true.view(), y_pred.view())?;
let r2 = metrics::regressor::r2_score(y_true.view(), y_pred.view())?;
println!("RMSE: {:.4}, R²: {:.4}", rmse, r2);
```

### 完整训练-评估流程

```rust
use naja::algorithms::linrg::LinearRegression;
use naja::core::traits::{SupervisedEstimator, Predictor};
use naja::metrics;
let model = LinearRegression::new().intercept(true);
let fitted = model.fit_supervised(x_train.view(), y_train.view())?;
let y_pred = fitted.predict(x_test.view())?;
let rmse = metrics::regressor::rmse(y_test.view(), y_pred.view())?;
let r2 = metrics::regressor::r2_score(y_test.view(), y_pred.view())?;
```

## 注意事项

- 所有指标函数校验 `y_true` 与 `y_pred` 长度一致
- 模块导出：`pub use regressor::*;`，可直接 `metrics::mse(...)` 调用
