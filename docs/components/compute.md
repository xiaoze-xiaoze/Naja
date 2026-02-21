# core::compute（数值运算原语）

## 概述

`core::compute` 负责提供稳定、可复用的数值运算原语，并统一后端选择（`ndarray` / `faer`）。算法实现只依赖 `Matrix/Vector` 与这些函数，不直接触碰后端细节。

## 模块位置

- `src/core/compute/types.rs`
- `src/core/compute/ops.rs`

## 类型定义

*   **Matrix / MatrixView / Vector / VectorView**
    *   **签名**: `Array2<f64>` / `ArrayView2<'_, f64>` / `Array1<f64>` / `ArrayView1<'_, f64>`
    *   **含义**: 统一矩阵与向量类型别名
    *   **用途**: 作为所有数值运算与算法接口的标准输入输出

## 函数分类

### 有限性与基础校验

*   **all_finite_vec / all_finite_mat**
    *   **签名**: `fn(v: VectorView<'_>) -> bool` / `fn(m: MatrixView<'_>) -> bool`
    *   **含义**: 检查向量/矩阵是否全为有限数
    *   **用途**: 防御性检查，避免 `NaN/Inf` 破坏计算

*   **ensure_nonempty_vec / ensure_nonempty_mat**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<()>` / `fn(m: MatrixView<'_>) -> Result<()>`
    *   **含义**: 检查向量/矩阵是否为空
    *   **用途**: 统一空输入的错误处理

*   **ensure_len**
    *   **签名**: `fn(a: VectorView<'_>, b: VectorView<'_>, name_a: &str, name_b: &str) -> Result<()>`
    *   **含义**: 校验两个向量长度是否一致
    *   **用途**: 二元向量运算的形状守卫

*   **ensure_matmul / ensure_gemv**
    *   **签名**: `fn(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<()>` / `fn(a: MatrixView<'_>, x: VectorView<'_>) -> Result<()>`
    *   **含义**: 校验矩阵乘法或矩阵向量乘法维度兼容
    *   **用途**: 线性代数计算前的形状检查

### 设计矩阵与正则化

*   **add_intercept**
    *   **签名**: `fn(x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 添加截距列形成设计矩阵
    *   **用途**: 线性/逻辑回归的截距处理

*   **add_diag_mut**
    *   **签名**: `fn(a: &mut Matrix, alpha: f64) -> Result<()>`
    *   **含义**: 原地对角加 `alpha`
    *   **用途**: L2 正则化或数值稳定化

### 向量运算

*   **dot**
    *   **签名**: `fn(a: VectorView<'_>, b: VectorView<'_>) -> Result<f64>`
    *   **含义**: 向量点积
    *   **用途**: 相似度、投影、梯度计算

*   **l2 / l2_sq**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<f64>`
    *   **含义**: 计算向量 L2 范数或平方
    *   **用途**: 距离与正则项计算

*   **add_scaled_mut**
    *   **签名**: `fn(dst: &mut Vector, src: VectorView<'_>, alpha: f64) -> Result<()>`
    *   **含义**: 执行 `dst += alpha * src`
    *   **用途**: 梯度更新的基础操作

### 矩阵与线代原语

*   **matmul**
    *   **签名**: `fn(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 矩阵乘法
    *   **用途**: 通用线性变换

*   **gemv**
    *   **签名**: `fn(a: MatrixView<'_>, x: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 矩阵-向量乘法
    *   **用途**: 线性模型推理

*   **xtx / xty**
    *   **签名**: `fn(x: MatrixView<'_>) -> Result<Matrix>` / `fn(x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 计算 $X^T X$ 或 $X^T y$
    *   **用途**: 最小二乘与正规方程中间量

### 列统计与预处理

*   **col_mean**
    *   **签名**: `fn(x: MatrixView<'_>) -> Result<Vector>`
    *   **含义**: 计算列均值
    *   **用途**: 标准化、中心化

*   **col_var**
    *   **签名**: `fn(x: MatrixView<'_>, ddof: usize) -> Result<Vector>`
    *   **含义**: 计算列方差，`ddof` 为自由度修正
    *   **用途**: 标准化、方差估计

*   **center_cols_mut / scale_cols_mut**
    *   **签名**: `fn(x: &mut Matrix, mean: VectorView<'_>) -> Result<()>` / `fn(x: &mut Matrix, scale: VectorView<'_>) -> Result<()>`
    *   **含义**: 原地对列做减均值或缩放
    *   **用途**: 数据预处理

### 概率与数值稳定

*   **sigmoid**
    *   **签名**: `fn(x: f64) -> f64`
    *   **含义**: Logistic Sigmoid
    *   **用途**: 概率映射

*   **log1pexp**
    *   **签名**: `fn(x: f64) -> f64`
    *   **含义**: 稳定计算 $\log(1 + e^x)$
    *   **用途**: Logistic 损失

*   **logsumexp**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<f64>`
    *   **含义**: 稳定计算 $\log(\sum e^{v_i})$
    *   **用途**: Softmax 分母

*   **softmax_mut**
    *   **签名**: `fn(v: &mut Vector) -> Result<()>`
    *   **含义**: 原地 Softmax
    *   **用途**: 多分类概率输出

*   **argmax**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<usize>`
    *   **含义**: 最大值索引
    *   **用途**: 分类决策

### 线性求解（faer-backend）

*   **solve_lu**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: LU 分解求解方阵线性系统
    *   **用途**: 通用线性方程组求解

*   **solve_cholesky**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: Cholesky 分解求解对称正定系统
    *   **用途**: 正规方程求解

*   **solve_lstsq**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 最小二乘解（QR 分解）
    *   **用途**: 直接求解线性回归

## 使用示例

```rust
use naja::core::compute::ops;

let mean = ops::col_mean(x.view())?;
let xtx = ops::xtx(x.view())?;
let y_pred = ops::gemv(x.view(), w.view())?;
```

## 注意事项

- 所有函数遵循 `core::Result` 的错误体系
- 线性求解函数依赖 `faer-backend` feature
