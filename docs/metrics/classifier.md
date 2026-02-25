# Classifier Metrics（分类指标）

分类评估指标，所有函数统一返回 `core::Result<f64>` 并进行输入校验。

## 模块位置

- `src/metrics/classifier.rs`

## 指标列表

| 函数 | 签名 | 含义 |
|------|------|------|
| `accuracy` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>) -> Result<f64>` | 预测正确的比例 |
| `precision` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | 预测为正中真实为正的比例 |
| `recall` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | 真实为正中被正确识别的比例 |
| `f1_score` | `fn(y_true: VectorView<'_>, y_pred: VectorView<'_>, pos_label: f64) -> Result<f64>` | Precision 与 Recall 调和平均 |

## 使用建议

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| Accuracy | 直观 | 类别均衡 |
| Precision | 关注误报 | 垃圾邮件检测（避免误判正常邮件） |
| Recall | 关注漏报 | 疾病筛查（避免漏诊） |
| F1 | 平衡 Precision/Recall | 类别不平衡 |

## 使用示例

```rust
use naja::metrics;
let acc = metrics::classifier::accuracy(y_true.view(), y_pred.view())?;
let prec = metrics::classifier::precision(y_true.view(), y_pred.view(), 1.0)?;
let rec = metrics::classifier::recall(y_true.view(), y_pred.view(), 1.0)?;
let f1 = metrics::classifier::f1_score(y_true.view(), y_pred.view(), 1.0)?;
println!("Accuracy: {:.2}%, F1: {:.4}", acc * 100.0, f1);
```

## 注意事项

- 所有指标函数校验 `y_true` 与 `y_pred` 长度一致
- `precision` / `recall` / `f1_score` 的 `pos_label` 参数指定正类标签值
- 模块导出：`pub use classifier::*;`，可直接 `metrics::accuracy(...)` 调用
