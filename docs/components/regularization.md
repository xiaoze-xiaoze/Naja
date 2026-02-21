# core::regularization（正则化模块）

## 概述

`core::regularization` 通过枚举 `Penalty` 提供统一正则化接口，支持 None、Ridge (L2)、Lasso (L1)。

## 模块位置

- `src/core/regularization.rs`

## 类型定义

*   **Penalty**
    *   **变体**:
        - `None`
        - `Ridge { alpha: f64 }`
        - `Lasso { alpha: f64 }`
    *   **用途**: 统一描述正则化策略

## 正则化类型

*   **Penalty::None**
    *   **含义**: 不添加正则化项
    *   **用途**: 数据量充足或不需要约束时

*   **Penalty::Ridge**
    *   **公式**: $$ R(w) = \frac{1}{2} \alpha ||w||_2^2 $$
    *   **含义**: 对权重平方和惩罚
    *   **用途**: 抑制过拟合、缓解共线性

*   **Penalty::Lasso**
    *   **公式**: $$ R(w) = \alpha ||w||_1 $$
    *   **含义**: 对权重绝对值和惩罚
    *   **用途**: 特征选择与稀疏解

## 核心方法

*   **loss**
    *   **签名**: `fn loss(&self, w: ArrayView1<f64>) -> f64`
    *   **含义**: 计算正则化损失
    *   **用途**: 组合进整体损失函数

*   **gradient**
    *   **签名**: `fn gradient(&self, w: ArrayView1<f64>) -> Array1<f64>`
    *   **含义**: 计算正则化梯度
    *   **用途**: 梯度更新

*   **apply_l2**
    *   **签名**: `fn apply_l2(&self, xtx: &mut Array2<f64>, intercept: bool)`
    *   **含义**: 将 L2 正则项加到 $X^T X$ 对角
    *   **用途**: 闭式解的 Ridge 修正

*   **apply_l1**
    *   **签名**: `fn apply_l1(&self, z: f64) -> f64`
    *   **含义**: L1 软阈值算子
    *   **用途**: 坐标下降中的 Lasso 更新

## 使用示例

```rust
use naja::core::regularization::Penalty;

let none = Penalty::None;
let ridge = Penalty::Ridge { alpha: 0.1 };
let lasso = Penalty::Lasso { alpha: 0.1 };
```

## 注意事项

- `apply_l2` 会根据 `intercept` 决定是否跳过截距项
- `apply_l1` 只对 `Penalty::Lasso` 生效
