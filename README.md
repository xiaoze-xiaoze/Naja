# Naja (Rust ML Library)

Naja 是一个面向工程使用的 Rust 机器学习库，目标是在保持 Rust 风格与可维护性的前提下，为常见算法提供一致、可组合的 API。

## API 设计（模型与求解解耦）

Naja 对外暴露的算法 API 采用统一的“三段式”范式：

1. Model：只描述“我要解决的问题是什么”，不携带训练后的参数
2. Solver：只描述“怎么解”，例如 Auto / closed-form / gradient descent
3. Solution：只描述“解出来的结果”，用于后续 predict/transform/导出参数等

这套范式的目标是让不同算法在使用方式上高度一致：用户主要记住 builder/build 与 solve/predict/transform 这几个动作，剩下的差异尽可能收敛到模型超参和输出形状上。

### 输入与输出约定

- 输入矩阵/向量以 `ndarray` 为基础：`Matrix = Array2<f64>`，`Vector = Array1<f64>`
- 所有公共 API 返回 `core::Result<T>`，统一错误类型与形状/数值校验策略

### 监督学习（回归/分类）示例

监督学习的输入通常是 `(x, y)`，求解得到 `Solution`，再用 `predict` 得到输出。

```rust
let model = LinearRegression::builder()
    .fit_intercept(true)
    .penalty(Penalty::Ridge { alpha: 1e-2 })
    .build();

let solution = solve(&model, &x_train, &y_train, Solver::Auto)?;
let y_pred = predict(&solution, &x_test)?;
```

分类问题同样遵循该范式（只是在输出语义上不同，例如 `predict_proba` / `predict`），接口形态保持一致：

```rust
let model = LogisticRegression::builder()
    .penalty(Penalty::L2 { alpha: 1e-2 })
    .build();

let solution = solve(&model, &x_train, &y_train, Solver::Auto)?;
let proba = predict_proba(&solution, &x_test)?;
let y_hat = predict(&solution, &x_test)?;
```

### 非监督学习（聚类）示例

非监督学习通常只有 `x`。输出一般是聚类中心、标签、密度结构等，仍然用 `solve -> Solution` 的流程保持一致。

```rust
let model = KMeans::builder()
    .k(8)
    .max_iter(300)
    .build();

let solution = solve(&model, &x, Solver::Auto)?;
let labels = predict(&solution, &x)?;
```

### 预处理/变换（Transformer）示例

Transformer 的核心是“学习一组变换参数”并用于 `transform`。同样可以按 `solve -> Solution -> transform` 组织：

```rust
let model = StandardScaler::builder().build();

let solution = solve(&model, &x_train, Solver::Auto)?;
let x_train_t = transform(&solution, &x_train)?;
let x_test_t = transform(&solution, &x_test)?;
```

### 命名与职责边界（建议长期稳定）

- `Algo::builder() ... .build()`：只构造 Model（纯超参/纯描述）
- `solve(&model, ...) -> Solution`：只负责求解/训练
- `predict(&solution, ...)` / `transform(&solution, ...)`：只负责推理或变换

后续如果需要训练过程信息（迭代次数、是否收敛、最终 loss 等），建议以并行的 `report` 形式扩展，而不是把报告字段塞进 Model 或 Solution，避免职责混淆与破坏 API 一致性。

## 算法目录

下面这 12 个尽量做到覆盖面均衡：

1. Linear Regression（回归；支持 OLS/Ridge/Lasso/ElasticNet 等 penalty 选项）
2. Logistic Regression（分类；支持 L2/L1/ElasticNet 等 penalty 选项）
3. SVM（线性 SVM 起步；核函数作为扩展点）
4. KNN（分类/回归）
5. Naive Bayes（Gaussian/Multinomial/Bernoulli 作为同一算法的不同变体）
6. Decision Tree（CART：分类/回归）
7. Random Forest（分类/回归）
8. XGBoost（先做 “XGBoost-style tree boosting” 子集：正则、学习率、子采样、列采样）
9. KMeans（聚类）
10. DBSCAN（密度聚类）
11. Gaussian Mixture Model（GMM/EM；概率聚类与密度估计）
12. LDA（Linear Discriminant Analysis：降维/分类，补足 PCA 之外的第二种降维路径）

## 项目结构

```
Naja/
├── Cargo.toml                         # crate 元信息与依赖
├── src/
│   ├── lib.rs                         # crate 入口；只做模块声明与公开导出
│   ├── core/                          # 仅为算法服务的最小公共层（小而稳定）
│   │   ├── mod.rs                     # core 入口；集中 re-export（给 algorithms 用）
│   │   ├── error.rs                   # 错误类型与 Result
│   │   ├── traits.rs                  # fit/predict/transform 等最小 trait
│   │   ├── data/                      # Dataset + 基础数据校验工具
│   │   │   ├── mod.rs
│   │   │   ├── dataset.rs
│   │   │   └── validate.rs
│   │   └── compute/                   # ndarray/faer 的统一适配入口（不写数值算法）
│   │       ├── mod.rs
│   │       ├── types.rs
│   │       └── ops.rs
│   ├── preprocessing/                 # 标准化/归一化等最小 transformer
│   │   ├── mod.rs
│   ├── metrics/                       # 指标：分类/回归/聚类
│   │   ├── mod.rs                     # metrics 模块入口
│   │   ├── classification.rs          # 分类指标
│   │   ├── regression.rs              # 回归指标
│   │   └── clustering.rs              # 聚类指标
│   └── algorithms/                    # 算法实现
│       ├── mod.rs                     # algorithms 模块入口
│       ├── linr.rs                    # Linear Regression
│       └── ...
├── examples/
└── tests/
```

## core::compute（ops.rs 运算清单）

`core::compute` 的目标是给所有算法提供一致、可复用的“数值运算原语”，并把后端选择（`ndarray` / `faer`）收敛在一处：算法实现尽量只依赖 `Matrix/Vector` 与这些函数，不直接碰后端细节。

下面列出建议长期稳定的一组 `ops.rs` 对外函数（按职责分组）。所有函数均应返回 `core::Result<T>`（或 `bool`），并使用 `core::Error` 统一形状与数值校验策略。

### 有限性与基础校验

- `is_finite_vec(v: VectorView<'_>) -> bool`：检查向量是否全为有限数（无 NaN/Inf）
- `is_finite_mat(m: MatrixView<'_>) -> bool`：检查矩阵是否全为有限数（无 NaN/Inf）
- `check_nonempty_vec(v: VectorView<'_>) -> Result<()>`：检查向量非空（空则 `EmptyInput`）
- `check_nonempty_mat(m: MatrixView<'_>) -> Result<()>`：检查矩阵行列均非零（零则 `EmptyInput`）
- `check_vec_len_eq(a, b, name_a, name_b) -> Result<()>`：向量长度一致性校验（不一致则 `InvalidShape`）
- `check_matmul_dims(a, b) -> Result<()>`：矩阵乘维度校验（`a.ncols == b.nrows`）
- `check_gemv_dims(a, x) -> Result<()>`：矩阵-向量乘维度校验（`a.ncols == x.len`）

### 设计矩阵与正则化

- `with_intercept(x: MatrixView<'_>) -> Result<Matrix>`：构造带截距项的设计矩阵（在最左侧添加全 1 列）
- `add_diag_inplace(a: &mut Matrix, alpha: f64) -> Result<()>`：原地对角线加法（常用于 Ridge/L2 正则或数值稳定）

### 向量运算

- `vec_dot(a: VectorView<'_>, b: VectorView<'_>) -> Result<f64>`：点积（含长度校验）
- `vec_l2_sq(v: VectorView<'_>) -> Result<f64>`：L2 范数平方（含非空校验）
- `vec_l2(v: VectorView<'_>) -> Result<f64>`：L2 范数
- `vec_add_scaled_inplace(dst: &mut Vector, src: VectorView<'_>, alpha: f64) -> Result<()>`：`dst += alpha * src`（用于梯度更新等）

### 矩阵与线代原语

- `matmul(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Matrix>`：矩阵乘（内部可按 feature 选择后端）
- `gemv(a: MatrixView<'_>, x: VectorView<'_>) -> Result<Vector>`：矩阵-向量乘
- `xtx(x: MatrixView<'_>) -> Result<Matrix>`：`X^T X`
- `xty(x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector>`：`X^T y`

### 列统计与预处理工具

- `col_mean(x: MatrixView<'_>) -> Result<Vector>`：按列均值（用于中心化/标准化）
- `col_var(x: MatrixView<'_>, ddof: usize) -> Result<Vector>`：按列方差（支持 `ddof`）
- `center_cols_inplace(x: &mut Matrix, mean: VectorView<'_>) -> Result<()>`：按列减均值（原地中心化）
- `scale_cols_inplace(x: &mut Matrix, scale: VectorView<'_>) -> Result<()>`：按列除尺度（原地缩放；拒绝 0 或非有限尺度）

### 概率/分类数值稳定函数

- `sigmoid(x: f64) -> f64`：数值稳定的 logistic sigmoid
- `log1pexp(x: f64) -> f64`：数值稳定计算 `log(1 + exp(x))`
- `logsumexp(v: VectorView<'_>) -> Result<f64>`：数值稳定计算 `log(sum(exp(v)))`
- `softmax_inplace(v: &mut Vector) -> Result<()>`：对向量原地 softmax（内部用 logsumexp 稳定）
- `argmax(v: VectorView<'_>) -> Result<usize>`：返回最大元素的下标（含非空校验）

### 线性求解（可选 faer 后端）

这些函数用于把“求解线性系统/最小二乘”从算法逻辑中剥离出来。若未开启 `faer-backend` feature，建议返回 `InvalidParam`，提示用户开启特性。

- `solve_square_lu(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`：方阵线性系统求解（LU with pivoting）
- `solve_spd_cholesky(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`：对称正定系统求解（Cholesky/LLT）
- `solve_lstsq_qr(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`：过定约最小二乘（QR，返回最小二乘解）
