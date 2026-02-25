# Clusterer Metrics（聚类指标）

聚类评估指标，所有函数统一返回 `core::Result<f64>` 并进行输入校验。

## 模块位置

- `src/metrics/clusterer.rs`

## 指标列表

| 函数 | 签名 | 公式 | 含义 |
|------|------|------|------|
| `silhouette_score` | `fn(x: MatrixView<'_>, labels: VectorView<'_>) -> Result<f64>` | $s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$ | 簇内紧密度与簇间分离度 |

- $a(i)$：样本 $i$ 到同簇其他点的平均距离
- $b(i)$：样本 $i$ 到最近其他簇的平均距离
- 取值范围 $[-1, 1]$，越大越好

## 使用示例

```rust
use naja::metrics;
let sil = metrics::clusterer::silhouette_score(x.view(), labels.view())?;
println!("Silhouette Score: {:.4}", sil);
```

## 注意事项

- `silhouette_score` 要求至少有 2 个簇且每簇至少 2 个样本
- 模块导出：`pub use clusterer::*;`，可直接 `metrics::silhouette_score(...)` 调用
