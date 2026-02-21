# metrics（指标库）

## 概述

指标模块用于评估模型性能，所有函数统一返回 `core::Result<f64>` 并进行输入校验。

## 模块位置

- `src/metrics/regressor.rs`
- `src/metrics/classifier.rs`
- `src/metrics/clusterer.rs`

## 指标分类

### 回归指标

*   **MSE**
    *   **签名**: `fn mse(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>`
    *   **公式**: $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    *   **含义**: 预测误差的平方均值
    *   **用途**: 对大误差敏感的回归评价

*   **RMSE**
    *   **签名**: `fn rmse(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>`
    *   **公式**: $$ \text{RMSE} = \sqrt{\text{MSE}} $$
    *   **含义**: 与原变量同量纲的误差
    *   **用途**: 直观比较不同模型的误差

*   **MAE**
    *   **签名**: `fn mae(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>`
    *   **公式**: $$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
    *   **含义**: 绝对误差均值
    *   **用途**: 对异常值不如 MSE 敏感

*   **R2 Score**
    *   **签名**: `fn r2_score(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>`
    *   **公式**: $$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
    *   **含义**: 解释方差的比例
    *   **用途**: 拟合优度评价

### 分类指标

*   **Accuracy**
    *   **签名**: `fn accuracy(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>`
    *   **含义**: 预测正确的比例
    *   **用途**: 最直观的分类指标

*   **Precision**
    *   **签名**: `fn precision(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>`
    *   **含义**: 预测为正类的样本中真实为正的比例
    *   **用途**: 关注误报成本的场景

*   **Recall**
    *   **签名**: `fn recall(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>`
    *   **含义**: 真实正类被正确识别的比例
    *   **用途**: 关注漏报成本的场景

*   **F1 Score**
    *   **签名**: `fn f1_score(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>`
    *   **含义**: Precision 与 Recall 的调和平均
    *   **用途**: 类别不平衡时的综合评价

### 聚类指标

*   **Silhouette Score**
    *   **签名**: `fn silhouette_score(x: MatrixView<'_>, labels: VectorView<'_>) -> Result<f64>`
    *   **公式**: $$ s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} $$
    *   **含义**: 衡量簇内紧密度与簇间分离度
    *   **用途**: 评估聚类质量

## 使用示例

```rust
use naja::metrics;

let mse = metrics::regressor::mse(y_true.view(), y_pred.view())?;
let acc = metrics::classifier::accuracy(y_true.view(), y_pred.view())?;
let sil = metrics::clusterer::silhouette_score(x.view(), labels.view())?;
```
