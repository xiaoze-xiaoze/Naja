# metrics

## 概述

`metrics` 模块提供模型性能评估指标，所有函数统一返回 `core::Result<f64>` 并进行输入校验。

## 模块位置

- `src/metrics/mod.rs`
- `src/metrics/regressor.rs`
- `src/metrics/classifier.rs`
- `src/metrics/clusterer.rs`

---

## 回归指标

| 函数 | 签名 | 公式 | 含义 |
|------|------|------|------|
| `mse` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 均方误差 |
| `rmse` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\sqrt{\text{MSE}}$ | 均方根误差 |
| `mae` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 平均绝对误差 |
| `r2_score` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | 决定系数 |

### 使用建议

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| MSE | 对大误差敏感 | 需要惩罚大偏差 |
| RMSE | 与原数据同量纲 | 直观解释误差大小 |
| MAE | 对异常值不敏感 | 数据含异常值 |
| R² | 无量纲，可比较 | 模型对比、解释方差比例 |

---

## 分类指标

| 函数 | 签名 | 含义 |
|------|------|------|
| `accuracy` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | 预测正确的比例 |
| `precision` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | 预测为正中真实为正的比例 |
| `recall` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | 真实为正中被正确识别的比例 |
| `f1_score` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | Precision 与 Recall 调和平均 |

### 使用建议

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| Accuracy | 直观 | 类别均衡 |
| Precision | 关注误报 | 垃圾邮件检测（避免误判正常邮件） |
| Recall | 关注漏报 | 疾病筛查（避免漏诊） |
| F1 | 平衡 Precision/Recall | 类别不平衡 |

---

## 聚类指标

| 函数 | 签名 | 公式 | 含义 |
|------|------|------|------|
| `silhouette_score` | `fn(x: MatrixView<'_>, labels: VectorView<'_>) -> Result<f64>` | $s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$ | 簇内紧密度与簇间分离度 |

- $a(i)$：样本 $i$ 到同簇其他点的平均距离
- $b(i)$：样本 $i$ 到最近其他簇的平均距离
- 取值范围 $[-1, 1]$，越大越好

---

## 使用示例

### 回归评估

```rust
use naja::metrics;

let mse = metrics::regressor::mse(y_true.view(), y_pred.view())?;
let rmse = metrics::regressor::rmse(y_true.view(), y_pred.view())?;
let r2 = metrics::regressor::r2_score(y_true.view(), y_pred.view())?;

println!("RMSE: {:.4}, R²: {:.4}", rmse, r2);
```

### 分类评估

```rust
use naja::metrics;

let acc = metrics::classifier::accuracy(y_true.view(), y_pred.view())?;
let prec = metrics::classifier::precision(y_true.view(), y_pred.view(), 1.0)?;
let rec = metrics::classifier::recall(y_true.view(), y_pred.view(), 1.0)?;
let f1 = metrics::classifier::f1_score(y_true.view(), y_pred.view(), 1.0)?;

println!("Accuracy: {:.2}%, F1: {:.4}", acc * 100.0, f1);
```

### 聚类评估

```rust
use naja::metrics;

let sil = metrics::clusterer::silhouette_score(x.view(), labels.view())?;
println!("Silhouette Score: {:.4}", sil);
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

---

## 注意事项

- 所有指标函数校验 `y_true` 与 `y_pred` 长度一致
- `precision` / `recall` / `f1_score` 的 `pos_label` 参数指定正类标签值
- `silhouette_score` 要求至少有 2 个簇且每簇至少 2 个样本
- 模块导出：`pub use regressor::*; pub use classifier::*; pub use clusterer::*;`，可直接 `metrics::mse(...)` 调用
