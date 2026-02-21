# Core Traits（核心抽象）

## 概述

核心 trait 定义了模型的训练、预测与变换接口，保证算法统一范式与可组合性。

## 模块位置

- `src/core/traits.rs`

## Trait 总览

| Trait | 方法 | 用途 |
|-------|------|------|
| `Predictor` | `predict(&self, x) -> Result<Vector>` | 预测输出 |
| `ProbabilisticPredictor` | `predict_proba(&self, x) -> Result<Matrix>` | 预测概率分布 |
| `Transformer` | `transform(&self, x) -> Result<Matrix>` | 数据变换 |
| `InverseTransformer` | `inverse_transform(&self, x) -> Result<Matrix>` | 逆变换 |
| `FitSupervised` | `fit(&self, x, y) -> Result<Object>` | 监督学习训练 |
| `FitUnsupervised` | `fit(&self, x) -> Result<Object>` | 无监督学习训练 |
| `PartialFit` | `partial_fit(&mut self, x, y) -> Result<()>` | 增量/在线学习 |

## Trait 说明

*   **FitSupervised**
    *   **签名**: `fn fit(&self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<Self::Object>`
    *   **含义**: 监督学习训练接口
    *   **用途**: 回归与分类模型的统一训练入口

*   **FitUnsupervised**
    *   **签名**: `fn fit(&self, x: MatrixView<'_>) -> Result<Self::Object>`
    *   **含义**: 无监督学习训练接口
    *   **用途**: 聚类与降维模型的统一训练入口

*   **Predictor**
    *   **签名**: `fn predict(&self, x: MatrixView<'_>) -> Result<Vector>`
    *   **含义**: 生成预测结果
    *   **用途**: 监督学习推理

*   **ProbabilisticPredictor**
    *   **签名**: `fn predict_proba(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 输出概率分布
    *   **用途**: 分类器概率预测

*   **Transformer**
    *   **签名**: `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 数据变换接口
    *   **用途**: 预处理、降维等变换

*   **InverseTransformer**
    *   **签名**: `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 逆变换接口
    *   **用途**: 将变换后的数据恢复原尺度

*   **PartialFit**
    *   **签名**: `fn partial_fit(&mut self, x: MatrixView<'_>, y: VectorView<'_>) -> Result<()>`
    *   **含义**: 增量训练接口
    *   **用途**: 流式或在线学习

## 范式说明

| 阶段 | 接口 | 输入 | 输出 |
|------|------|------|------|
| Define | 构造函数 | 超参数 | `Model` |
| Fit | `FitSupervised` / `FitUnsupervised` | 数据 | `Solution` |
| Predict | `Predictor` / `ProbabilisticPredictor` | 新数据 | 预测结果 |
| Transform | `Transformer` / `InverseTransformer` | 数据 | 变换结果 |

## 标记 Trait

*   **SupervisedModel**
    *   **含义**: 标记监督学习模型

*   **UnsupervisedModel**
    *   **含义**: 标记无监督学习模型
