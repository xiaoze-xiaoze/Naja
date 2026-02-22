# preprocessing（数据预处理）

## 概述

`preprocessing` 提供标准化等预处理工具，当前核心类型为 `StandardScaler`，采用 typestate 区分拟合前后能力。

## 模块位置

- `src/preprocessing/mod.rs`
- `src/preprocessing/scaler.rs`

## 类型定义

*   **StandardScaler<Unfitted>**
    *   **含义**: 未拟合的标准化器
    *   **用途**: 仅允许 `fit`

*   **StandardScaler<Fitted>**
    *   **含义**: 已拟合的标准化器
    *   **用途**: 允许 `transform` 与 `inverse_transform`

*   **core::traits::Transformer / InverseTransformer**
    *   **签名**: `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>` / `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 统一的数据变换与逆变换接口
    *   **用途**: 预处理流水线

## 方法

*   **StandardScaler::new**
    *   **签名**: `fn new() -> StandardScaler<Unfitted>`
    *   **含义**: 创建未拟合实例
    *   **用途**: 作为预处理入口

*   **StandardScaler<Unfitted>::fit**
    *   **签名**: `fn fit(self, data: MatrixView<'_>) -> StandardScaler<Fitted>`
    *   **含义**: 计算 `mean/std`
    *   **用途**: 生成可变换的已拟合对象

*   **StandardScaler<Fitted>::transform**
    *   **签名**: `fn transform(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 标准化变换 $(x - \text{mean}) / \text{std}$
    *   **用途**: 训练集/测试集统一变换

*   **StandardScaler<Fitted>::inverse_transform**
    *   **签名**: `fn inverse_transform(&self, x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 逆变换 $x \times \text{std} + \text{mean}$
    *   **用途**: 从标准化空间恢复原尺度

## 使用示例

```rust
use naja::preprocessing::StandardScaler;
use naja::core::traits::{Transformer, InverseTransformer};

let scaler = StandardScaler::new();
let fitted = scaler.fit(x_train.view());
let x_train_scaled = fitted.transform(x_train.view())?;
let x_test_scaled = fitted.transform(x_test.view())?;
let x_train_back = fitted.inverse_transform(x_train_scaled.view())?;
```

## 注意事项

- 只用训练集 `fit`，再用拟合对象变换测试集
- `std == 0` 的列会被替换为 1，避免除零
- 预处理组件统一实现 `core::traits::Transformer` 与 `core::traits::InverseTransformer`
