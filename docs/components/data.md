# core::data（Dataset 与 validate）

## 概述

`core::data` 提供轻量数据容器 `Dataset` 与配套校验函数，适合算法示例与中小规模数据场景。

## 模块位置

- `src/core/data/dataset.rs`
- `src/core/data/validate.rs`

## 类型定义

*   **Dataset**
    *   **字段**:
        - `records: Matrix`
        - `targets: Vector`
        - `feature_names: Vec<String>`
    *   **含义**: 统一管理特征矩阵、目标向量与特征名
    *   **用途**: 为算法与示例提供一致的数据入口

## 方法

*   **Dataset::new**
    *   **签名**: `fn new(records: Matrix, targets: Vector) -> Result<Self>`
    *   **含义**: 构造数据集并校验维度
    *   **用途**: 标准数据集入口

*   **Dataset::with_feature_names**
    *   **签名**: `fn with_feature_names(self, names: Vec<String>) -> Result<Self>`
    *   **含义**: 设置特征名并校验数量
    *   **用途**: 提升可读性与可解释性

*   **Dataset::split**
    *   **签名**: `fn split(&self, test_ratio: f64) -> (Self, Self)`
    *   **含义**: 按顺序划分训练集/测试集
    *   **用途**: 快速切分数据

## 校验函数

*   **check_dimensions**
    *   **签名**: `fn(records: &Matrix, targets: &Vector) -> Result<()>`
    *   **含义**: 校验样本数一致性
    *   **错误**: `InvalidShape`

*   **check_feature**
    *   **签名**: `fn(n_features: usize, names: &[String]) -> Result<()>`
    *   **含义**: 校验特征数与特征名数量一致性
    *   **错误**: `InvalidShape`

## 使用示例

```rust
use naja::core::data::Dataset;

let dataset = Dataset::new(records, targets)?;
let dataset = dataset.with_feature_names(vec!["age".into(), "income".into()])?;
let (train, test) = dataset.split(0.2);
```

## 适用场景

- 快速原型与示例
- 中小规模、内存可容纳的数据
- 不需要复杂流水线的训练流程

## 注意事项

- `split` 按顺序切分，不进行随机打乱
