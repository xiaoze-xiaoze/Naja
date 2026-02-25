# core::compute（数值运算原语）

## 概述

`core::compute` 提供稳定、可复用的数值运算原语，统一后端选择（`ndarray` / `faer`）。算法实现只依赖 `Matrix/Vector` 与这些函数，不直接触碰后端细节。

## 模块位置

- `src/core/compute/types.rs`
- `src/core/compute/ops.rs`

## 类型定义

```rust
pub type Matrix = ndarray::Array2<f64>;
pub type Vector = ndarray::Array1<f64>;
pub type MatrixView<'a> = ndarray::ArrayView2<'a, f64>;
pub type VectorView<'a> = ndarray::ArrayView1<'a, f64>;
```

| 类型 | 含义 | 用途 |
|------|------|------|
| `Matrix` | `Array2<f64>` | 矩阵（拥有所有权） |
| `Vector` | `Array1<f64>` | 向量（拥有所有权） |
| `MatrixView<'a>` | `ArrayView2<'a, f64>` | 矩阵视图（零拷贝引用） |
| `VectorView<'a>` | `ArrayView1<'a, f64>` | 向量视图（零拷贝引用） |

---

## 函数分类

### 有限性与基础校验

| 函数 | 签名 | 含义 |
|------|------|------|
| `all_finite_vec` | `fn(v: VectorView<'_>) -> bool` | 检查向量是否全为有限数 |
| `all_finite_mat` | `fn(m: MatrixView<'_>) -> bool` | 检查矩阵是否全为有限数 |
| `ensure_nonempty_vec` | `fn(v: VectorView<'_>) -> Result<()>` | 检查向量是否为空 |
| `ensure_nonempty_mat` | `fn(m: MatrixView<'_>) -> Result<()>` | 检查矩阵是否为空 |
| `ensure_len` | `fn(a: VectorView<'_>, b: VectorView<'_>, name_a: &str, name_b: &str) -> Result<()>` | 校验两向量长度一致 |
| `ensure_matmul` | `fn(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<()>` | 校验矩阵乘法维度兼容 |
| `ensure_gemv` | `fn(a: MatrixView<'_>, x: VectorView<'_>) -> Result<()>` | 校验矩阵-向量乘法维度兼容 |

### 设计矩阵与正则化

| 函数 | 签名 | 含义 |
|------|------|------|
| `add_intercept` | `fn(x: MatrixView<'_>) -> Result<Matrix>` | 添加截距列（首列填 1）形成设计矩阵 |
| `add_diag_mut` | `fn(a: &mut Matrix, alpha: f64) -> Result<()>` | 原地对角加 `alpha`（L2 正则化） |

### 向量运算

| 函数 | 签名 | 含义 |
|------|------|------|
| `dot` | `fn(a: VectorView<'_>, b: VectorView<'_>) -> Result<f64>` | 向量点积 |
| `l2` | `fn(v: VectorView<'_>) -> Result<f64>` | L2 范数 $\|v\|_2$ |
| `l2_sq` | `fn(v: VectorView<'_>) -> Result<f64>` | L2 范数平方 $\|v\|_2^2$ |
| `add_scaled_mut` | `fn(dst: &mut Vector, src: VectorView<'_>, alpha: f64) -> Result<()>` | `dst += alpha * src` |

### 矩阵与线性代数

| 函数 | 签名 | 含义 |
|------|------|------|
| `matmul` | `fn(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Matrix>` | 矩阵乘法 $AB$ |
| `gemv` | `fn(a: MatrixView<'_>, x: VectorView<'_>) -> Result<Vector>` | 矩阵-向量乘法 $Ax$ |
| `xtx` | `fn(x: MatrixView<'_>) -> Result<Matrix>` | 计算 $X^T X$ |
| `xty` | `fn(x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector>` | 计算 $X^T y$ |

### 列统计与预处理

| 函数 | 签名 | 含义 |
|------|------|------|
| `col_mean` | `fn(x: MatrixView<'_>) -> Result<Vector>` | 计算每列均值 |
| `col_var` | `fn(x: MatrixView<'_>, ddof: usize) -> Result<Vector>` | 计算每列方差（`ddof` 为自由度修正） |
| `center_cols_mut` | `fn(x: &mut Matrix, mean: VectorView<'_>) -> Result<()>` | 原地列减均值 |
| `scale_cols_mut` | `fn(x: &mut Matrix, scale: VectorView<'_>) -> Result<()>` | 原地列除以缩放因子 |

### 概率与数值稳定

| 函数 | 签名 | 含义 |
|------|------|------|
| `sigmoid` | `fn(x: f64) -> f64` | Logistic Sigmoid $\frac{1}{1+e^{-x}}$ |
| `log1pexp` | `fn(x: f64) -> f64` | 稳定计算 $\log(1+e^x)$ |
| `logsumexp` | `fn(v: VectorView<'_>) -> Result<f64>` | 稳定计算 $\log(\sum e^{v_i})$ |
| `softmax_mut` | `fn(v: &mut Vector) -> Result<()>` | 原地 Softmax |
| `argmax` | `fn(v: VectorView<'_>) -> Result<usize>` | 返回最大值索引 |

### 线性求解器（需 `faer-backend` feature）

| 函数 | 签名 | 含义 | 适用场景 |
|------|------|------|----------|
| `solve_lu` | `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>` | LU 分解求解 $Ax=b$ | 通用方阵线性系统 |
| `solve_cholesky` | `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>` | Cholesky 分解求解 | 对称正定矩阵（如正规方程） |
| `solve_lstsq` | `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>` | 最小二乘解（QR 分解） | 超定系统（nrows >= ncols） |
| `solve_svd` | `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>` | SVD 分解求解 | 病态矩阵、奇异值过滤 |

---

## Feature 依赖

| Feature | 启用的函数 |
|---------|-----------|
| （默认） | 所有校验、向量运算、矩阵运算、概率函数 |
| `faer-backend` | `solve_lu`, `solve_cholesky`, `solve_lstsq`, `solve_svd` |

未启用 `faer-backend` 时调用线性求解器会返回 `BackendUnavailable` 错误。

---

## 使用示例

### 基础运算

```rust
use naja::core::compute::ops;
let mean = ops::col_mean(x.view())?;
let xtx = ops::xtx(x.view())?;
let y_pred = ops::gemv(x.view(), w.view())?;
```

### 线性回归正规方程

```rust
use naja::core::compute::ops;
let x_design = ops::add_intercept(x.view())?;
let xtx = ops::xtx(x_design.view())?;
let xty = ops::xty(x_design.view(), y.view())?;
let w = ops::solve_cholesky(xtx.view(), xty.view())?;
```

### Softmax 分类

```rust
use naja::core::compute::ops;
let mut logits = vec![1.0, 2.0, 3.0];
let mut v = ndarray::Array1::from_vec(logits);
ops::softmax_mut(&mut v)?;
let class = ops::argmax(v.view())?;
```

---

## 注意事项

- 所有函数返回 `core::Result<T>`，统一错误处理
- 线性求解器要求启用 `faer-backend` feature
- `solve_cholesky` 要求矩阵对称正定，否则返回 `LinAlg` 错误
- `solve_lstsq` 要求 `nrows >= ncols`（超定或方阵）
- `solve_svd` 对病态矩阵有更好的数值稳定性，但计算成本更高
