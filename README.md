# Naja (Rust ML Library)

Naja 是一个面向工程使用的 Rust 机器学习库，目标是在保持 Rust 风格与可维护性的前提下，为常见算法提供一致、可组合的 API。

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
│       ├── linreg.rs
│       ├── logreg.rs
│       ├── svm.rs
│       ├── knn.rs
│       ├── nbayes.rs
│       ├── dectree.rs
│       ├── rforest.rs
│       ├── xgboost.rs
│       ├── kmeans.rs
│       ├── dbscan.rs
│       ├── gmm.rs
│       └── lda.rs
└── examples/
```

## API 设计（4+1 阶段范式）

Naja 采用统一的"4+1 阶段"范式，每个阶段职责明确、类型安全：

| 阶段 | 方法 | 职责 | 输出类型 |
|------|------|------|----------|
| Define | `new()` | 创建模型实例 | `Model<T>` |
| Configure | `configure(ConfigArgs {...})` | 配置超参数 | `Spec<T>` |
| Solve | `solve(SolveArgs {...})` | 训练/求解 | `Solution<T>` |
| Predict | `predict(&x)` / `transform(&x)` | 推理/变换 | 预测结果 |
| Inspect | `report()` | 训练报告 | `Report` |

### 核心设计原则

1. **阶段显式**：每个阶段一个入口方法，参数封装在结构体中
2. **类型安全**：未 `solve` 的模型无法 `predict`（编译期保证）
3. **命名参数**：使用结构体初始化语法，参数含义一目了然
4. **变量统一**：可全程使用同一变量名（shadowing），或按阶段区分
5. **纯文本报告**：Inspect 阶段返回的 Report 包含关键统计量和格式化信息的对象，直接打印即可获得人类可读的分析报告。

### 输入与输出约定

- 输入矩阵/向量以 `ndarray` 为基础：`Matrix = Array2<f64>`，`Vector = Array1<f64>`
- 所有公共 API 返回 `core::Result<T>`，统一错误类型与形状/数值校验策略
- Inspect 返回的 Report 实现 `Display`，设计为纯文本输出，不包含任何绘图逻辑。

### 监督学习（回归）示例

```rust
// 阶段1: Define - 创建模型
let model = LinearRegression::new();

// 阶段2: Configure - 配置超参数
let spec = model.configure(ConfigArgs {
    intercept: true,
    penalty: Penalty::Ridge { alpha: 1e-2 },
});

// 阶段3: Solve - 训练求解
let solution = spec.solve(SolveArgs {
    x: &x_train,
    y: &y_train,
    solver: Solver::Auto,
})?;

// 阶段4: Predict - 推理
let y_pred = solution.predict(&x_test)?;

// 阶段5: Inspect - 查看训练报告
let report = solution.report()?;
println!("{}", report);
println!("R2 Score: {}", report.r2_score);
```

### 监督学习（分类）示例

```rust
let model = LogisticRegression::new();

let spec = model.configure(ConfigArgs {
    penalty: Penalty::L2 { alpha: 1e-2 },
});

let solution = spec.solve(SolveArgs {
    x: &x_train,
    y: &y_train,
    solver: Solver::Auto,
})?;

let proba = solution.predict_proba(&x_test)?;
let y_hat = solution.predict(&x_test)?;
```

### 非监督学习（聚类）示例

```rust
let model = KMeans::new();

let spec = model.configure(ConfigArgs {
    k: 8,
    max_iter: 300,
});

let solution = spec.solve(SolveArgs {
    x: &x,
    solver: Solver::Auto,
})?;

let labels = solution.predict(&x)?;
```

### Transformer 示例

```rust
let model = StandardScaler::new();

let spec = model.configure(ConfigArgs {
    with_mean: true,
    with_std: true,
});

let solution = spec.solve(SolveArgs {
    x: &x_train,
});

let x_train_scaled = solution.transform(&x_train)?;
let x_test_scaled = solution.transform(&x_test)?;
```

### 简写形式（熟练用户）

对于简单场景，可利用链式调用和 shadowing 压缩代码：

```rust
let model = LinearRegression::new()
    .configure(ConfigArgs { intercept: true, penalty: Penalty::None });

let solution = model.solve(SolveArgs { x: &x_train, y: &y_train, solver: Solver::Auto })?;

let y_pred = solution.predict(&x_test)?;
```

### 命名与职责边界（长期稳定）

- `Model::new()`：创建默认实例，不携带任何配置
- `configure(ConfigArgs {...})`：纯配置阶段，返回 `Spec<T>`
- `solve(SolveArgs {...})`：纯训练阶段，返回 `Solution<T>`
- `predict(&x)` / `transform(&x)`：纯推理阶段
- `report()`：查看训练报告（如迭代次数、收敛情况、最终 Loss 等），具体格式待定。

## core::compute（ops.rs 运算清单）

`core::compute` 的目标是给所有算法提供一致、可复用的“数值运算原语”，并把后端选择（`ndarray` / `faer`）收敛在一处：算法实现尽量只依赖 `Matrix/Vector` 与这些函数，不直接碰后端细节。

下面列出建议长期稳定的一组 `ops.rs` 对外函数。所有函数均应返回 `core::Result<T>`（或 `bool`），并使用 `core::Error` 统一形状与数值校验策略。

### 有限性与基础校验

*   **all_finite_vec / all_finite_mat**
    *   **签名**: `fn(v: VectorView<'_>) -> bool` / `fn(m: MatrixView<'_>) -> bool`
    *   **含义**: 检查向量/矩阵是否全为有限数（无 `NaN` 或 `Inf`）。
    *   **用途**: 算法入口处的防御性检查，防止脏数据导致计算崩溃。

*   **ensure_nonempty_vec / ensure_nonempty_mat**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<()>`
    *   **含义**: 检查向量/矩阵是否为空。
    *   **用途**: 拒绝空输入（返回 `EmptyInput` 错误）。

*   **ensure_len**
    *   **签名**: `fn(a, b, name_a, name_b) -> Result<()>`
    *   **含义**: 校验两个向量长度是否一致。
    *   **用途**: 二元运算（如点积、加法）前的形状检查（不一致返回 `InvalidShape`）。

*   **ensure_matmul / ensure_gemv**
    *   **签名**: `fn(a, b) -> Result<()>` / `fn(a, x) -> Result<()>`
    *   **含义**: 校验矩阵乘法 (`A * B`) 或矩阵向量乘法 (`A * x`) 的维度兼容性。
    *   **用途**: 线性代数运算前的维度守卫（`A.cols == B.rows`）。

### 设计矩阵与正则化

*   **add_intercept**
    *   **签名**: `fn(x: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 构造带截距项的设计矩阵（在最左侧添加全 1 列）。
    *   **用途**: 线性模型（Linear/Logistic Regression）处理截距项的标准做法。

*   **add_diag_mut**
    *   **签名**: `fn(a: &mut Matrix, alpha: f64) -> Result<()>`
    *   **含义**: 原地将标量 `alpha` 加到矩阵对角线上。
    *   **用途**: 实现 Ridge (L2) 正则化，或增加矩阵的数值稳定性（防止奇异）。

### 向量运算

*   **dot**
    *   **签名**: `fn(a: VectorView<'_>, b: VectorView<'_>) -> Result<f64>`
    *   **含义**: 计算两个向量的点积。
    *   **用途**: 投影、相似度计算等基础运算。

*   **l2 / l2_sq**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<f64>`
    *   **含义**: 计算向量的 L2 范数（欧几里得范数）或其平方。
    *   **用途**: 距离计算、正则化项计算。

*   **add_scaled_mut**
    *   **签名**: `fn(dst: &mut Vector, src: VectorView<'_>, alpha: f64) -> Result<()>`
    *   **含义**: 执行 `dst += alpha * src` 操作。
    *   **用途**: 梯度下降更新参数的核心步骤（AXPY 操作）。

### 矩阵与线代原语

*   **matmul**
    *   **签名**: `fn(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Matrix>`
    *   **含义**: 矩阵乘法。
    *   **用途**: 核心变换操作，内部可按 feature 自动选择最佳后端。

*   **gemv**
    *   **签名**: `fn(a: MatrixView<'_>, x: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 矩阵-向量乘法。
    *   **用途**: 线性模型的前向传播（推理）。

*   **xtx / xty**
    *   **签名**: `fn(x: MatrixView<'_>) -> Result<Matrix>` / `fn(...) -> Result<Vector>`
    *   **含义**: 计算 $X^T X$ 或 $X^T y$。
    *   **用途**: 最小二乘法（Normal Equation）的中间步骤，利用对称性优化计算。

### 列统计与预处理工具

*   **col_mean / col_var**
    *   **签名**: `fn(...) -> Result<Vector>`
    *   **含义**: 计算矩阵各列的均值或方差（支持 `ddof`）。
    *   **用途**: 数据标准化（StandardScaler）的基础统计量。

*   **center_cols_mut / scale_cols_mut**
    *   **签名**: `fn(...) -> Result<()>`
    *   **含义**: 原地对矩阵各列减去均值 / 除以尺度。
    *   **用途**: 执行中心化或缩放操作。

### 概率/分类数值稳定函数

*   **sigmoid**
    *   **签名**: `fn(x: f64) -> f64`
    *   **含义**: Logistic Sigmoid 激活函数 $\frac{1}{1 + e^{-x}}$。
    *   **用途**: 将实数映射到 $(0, 1)$ 区间，用于概率预测。

*   **log1pexp**
    *   **签名**: `fn(x: f64) -> f64`
    *   **含义**: 数值稳定地计算 $\log(1 + e^x)$。
    *   **用途**: Logistic Regression 的损失函数计算（避免溢出）。

*   **logsumexp**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<f64>`
    *   **含义**: 数值稳定地计算 $\log(\sum e^{v_i})$。
    *   **用途**: Softmax 的分母计算，防止指数爆炸。

*   **softmax_mut**
    *   **签名**: `fn(v: &mut Vector) -> Result<()>`
    *   **含义**: 原地对向量应用 Softmax 变换。
    *   **用途**: 多分类问题的概率输出。

*   **argmax**
    *   **签名**: `fn(v: VectorView<'_>) -> Result<usize>`
    *   **含义**: 返回向量中最大值的索引。
    *   **用途**: 从概率分布中选取最可能的类别。

### 线性求解（可选 faer 后端）

这些函数用于把“求解线性系统/最小二乘”从算法逻辑中剥离出来。若未开启 `faer-backend` feature，建议返回 `InvalidParam`，提示用户开启特性。

*   **solve_lu**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 使用 LU 分解求解方阵线性系统 $Ax = b$。
    *   **用途**: 通用线性方程组求解。

*   **solve_cholesky**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 使用 Cholesky 分解求解对称正定系统 $Ax = b$。
    *   **用途**: 求解正规方程（Normal Equation），比 LU 更快更稳。

*   **solve_lstsq**
    *   **签名**: `fn(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector>`
    *   **含义**: 求解过定方程组的最小二乘解（QR 分解）。
    *   **用途**: 直接求解线性回归，无需构造 $X^T X$（数值上更稳定）。

## core::regularization（正则化模块）

`core::regularization` 提供了统一的正则化接口，用于在损失函数中添加惩罚项，以防止过拟合。通过枚举 `Penalty`，算法可以轻松支持 None, Ridge (L2), Lasso (L1) 等多种模式。

### 1. 正则化类型 (Penalty Types)

*   **Ridge (L2 Regularization)**
    *   **公式**: $$ R(w) = \frac{1}{2} \alpha ||w||_2^2 = \frac{1}{2} \alpha \sum w_j^2 $$
    *   **含义**: 对权重的平方和进行惩罚（L2 范数）。
    *   **用途**: 防止模型过拟合，处理多重共线性问题。倾向于使权重普遍变小，但不会变为 0。

*   **Lasso (L1 Regularization)**
    *   **公式**: $$ R(w) = \alpha ||w||_1 = \alpha \sum |w_j| $$
    *   **含义**: 对权重的绝对值和进行惩罚（L1 范数）。
    *   **用途**: 特征选择。倾向于产生稀疏解（Sparse Solution），即自动将不重要的特征权重置为 0。

### 2. 核心运算 (Core Operations)

*   **loss**
    *   **签名**: `fn loss(&self, w: ArrayView1<f64>) -> f64`
    *   **含义**: 计算当前权重向量 $w$ 对应的正则化损失值。
    *   **用途**: 计算总 Loss 时调用。

*   **gradient**
    *   **签名**: `fn gradient(&self, w: ArrayView1<f64>) -> Array1<f64>`
    *   **含义**: 计算正则化项关于权重 $w$ 的梯度。
    *   **公式**:
        *   Ridge: $\alpha \cdot w$
        *   Lasso: $\alpha \cdot \text{sign}(w)$
    *   **用途**: 梯度下降法 (Gradient Descent) 中更新权重。

*   **apply_l2**
    *   **签名**: `fn apply_l2(&self, xtx: &mut Array2<f64>, intercept: bool)`
    *   **含义**: 将 L2 正则项应用到 $X^T X$ 矩阵上（即 $X^T X + \alpha I$）。
    *   **细节**: 如果 `intercept` 为 true，会跳过第一个对角元素（不对截距项进行正则化）。
    *   **用途**: 闭式解 (Closed-form) 求解 Ridge Regression。

*   **apply_l1**
    *   **签名**: `fn apply_l1(&self, z: f64) -> f64`
    *   **含义**: 应用 L1 正则化的**软阈值算子 (Soft Thresholding Operator)**。
    *   **公式**: $S(z, \alpha) = \text{sign}(z) \cdot (|z| - \alpha)_+$
    *   **用途**: 坐标下降法 (Coordinate Descent) 求解 Lasso Regression。

## Metrics 指标库

位于 `src/metrics` 模块，提供了一系列标准的评估指标，用于衡量模型的性能。所有指标函数均经过输入校验（如维度一致性、非空检查），并统一返回 `core::Result<f64>`。

### 1. 回归指标 (Regression)

用于评估连续值预测模型的误差。

*   **Mean Squared Error (MSE)**
    *   **公式**: $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    *   **含义**: 预测值与真实值差的平方和的均值。
    *   **用途**: 最常用的回归损失函数，对大误差敏感（因为平方放大了误差）。

*   **Root Mean Squared Error (RMSE)**
    *   **公式**: $$ RMSE = \sqrt{MSE} $$
    *   **含义**: MSE 的平方根。
    *   **用途**: 量纲与原变量一致，比 MSE 更直观。

*   **Mean Absolute Error (MAE)**
    *   **公式**: $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
    *   **含义**: 预测值与真实值差的绝对值的均值。
    *   **用途**: 对异常值不如 MSE 敏感，更能反映预测误差的实际平均水平。

*   **R-squared ($R^2$) Score**
    *   **公式**: $$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
    *   **含义**: 决定系数，表示模型解释了数据方差的比例。
    *   **用途**: 衡量模型拟合优度，1 表示完美拟合，0 表示模型不如直接取均值，负数表示模型极差。

### 2. 分类指标 (Classification)

用于评估离散类别预测模型的准确性。目前主要支持二分类（需指定 `pos_label`）。

*   **Accuracy**
    *   **公式**: $$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
    *   **含义**: 预测正确的样本占总样本的比例。
    *   **用途**: 最直观的指标，但在样本不平衡时可能失效。

*   **Precision (查准率)**
    *   **公式**: $$ Precision = \frac{TP}{TP + FP} $$
    *   **含义**: 预测为正类的样本中，真正为正类的比例。
    *   **用途**: 关注“预测出的正例有多少是准的”，适用于对误报（False Positive）敏感的场景（如垃圾邮件过滤）。

*   **Recall (查全率/召回率)**
    *   **公式**: $$ Recall = \frac{TP}{TP + FN} $$
    *   **含义**: 所有真实正类样本中，被正确预测为正类的比例。
    *   **用途**: 关注“真实的正例有多少被找出来了”，适用于对漏报（False Negative）敏感的场景（如疾病诊断）。

*   **F1 Score**
    *   **公式**: $$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} $$
    *   **含义**: Precision 和 Recall 的调和平均数。
    *   **用途**: 综合衡量指标，适用于需要兼顾准确率和召回率，或样本不平衡的场景。

### 3. 聚类指标 (Clustering)

用于评估无监督聚类结果的质量。

*   **Silhouette Score (轮廓系数)**
    *   **公式**: 对每个样本 $i$，$$ s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} $$
        *   $a(i)$: **簇内紧密度**。样本 $i$ 到同簇其他样本的平均距离。越小越好。
        *   $b(i)$: **簇间分离度**。样本 $i$ 到最近异簇（与 $i$ 不在同簇的最近簇）所有样本的平均距离。越大越好。
        *   最终得分为所有样本 $s(i)$ 的均值。
    *   **含义**: 取值范围 $[-1, 1]$。
        *   **接近 1**: 样本 $i$ 距离同簇样本很近，距离异簇样本很远。聚类效果好。
        *   **接近 0**: 样本 $i$ 位于两个簇的边界上。聚类模糊。
        *   **接近 -1**: 样本 $i$ 距离同簇远，距离异簇近。样本可能被错误地分配到了当前簇。
    *   **用途**: 评估聚类的致密性和分离度，无需真实标签。常用于寻找 KMeans 中的最佳 $k$ 值（Elbow Method 的替代或补充）。
    *   **注意**: 计算复杂度为 $O(N^2)$，数据量较大时计算会非常慢。

## 线性回归 (Linear Regression)

线性回归是最基础的回归模型，用于预测连续目标变量 $y$ 与特征 $X$ 之间的线性关系。

### 数学模型
$$ y = Xw + b + \epsilon $$
其中 $w$ 是权重向量，$b$ 是截距（bias/intercept）。

### 配置参数 (ConfigArgs)

*   **intercept**: `bool` (默认 `true`)
    *   是否拟合截距项 $b$。
*   **penalty**: `Penalty` (默认 `None`)
    *   正则化类型：`None` (OLS), `Ridge`, `Lasso`。
*   **max_iter**: `usize` (默认 `1000`)
    *   最大迭代次数，仅用于 Lasso 求解。
*   **tol**: `f64` (默认 `1e-4`)
    *   收敛容差，仅用于 Lasso 求解。

### 求解策略 (Implementation Details)

根据 `penalty` 的不同，采用不同的求解器：

#### 1. 闭式解 (Closed-form Solver)
**适用场景**: `Penalty::None` (OLS) 和 `Penalty::Ridge`。
**原理**: 此时目标函数是二次凸函数，存在解析解。
$$ w = (X^T X + \alpha I)^{-1} X^T y $$
**实现步骤**:
1.  计算 $X^T X$ 和 $X^T y$。
2.  如果是 Ridge，向 $X^T X$ 的对角线加入 $\alpha$ (调用 `apply_l2`)。
3.  使用 Cholesky 分解求解线性方程组（比直接求逆更稳定）。

#### 2. 坐标下降法 (Coordinate Descent Solver)

**适用场景**: `Penalty::Lasso`。

**原理**: 由于 L1 正则项 $|w|_1$ 在零点不可导，无法使用梯度下降或闭式解。坐标下降法利用 L1 正则项的可分离性，每次固定其他维度，只优化一个 $w_j$。

**算法流程 (Algorithm Steps)**:

1.  **初始化**: 权重 $w = 0$，残差 $r = y$。
2.  **迭代优化**: 对每个特征 $j$ (从 1 到 $p$) 循环执行：
    *   **计算相关性 (Correlation)**: 计算特征 $x_j$ 与当前**部分残差 (Partial Residual)** 的内积 $\rho_j$。
        $$ \rho_j = x_j^T (y - \sum_{k \neq j} x_k w_k) = x_j^T r + w_j^{(old)} ||x_j||^2 $$
        *注意：代码实现中直接利用当前残差 $r$ 和旧权重 $w_j^{(old)}$ 快速计算，无需重新求和。*
    *   **软阈值更新 (Soft Thresholding)**:
        $$ w_j^{(new)} = \frac{S(\rho_j, \alpha)}{||x_j||^2} $$
        其中 $S(z, \alpha)$ 是软阈值算子：
        $$ S(z, \alpha) = \begin{cases} z - \alpha & \text{if } z > \alpha \\ z + \alpha & \text{if } z < -\alpha \\ 0 & \text{if } |z| \le \alpha \end{cases} $$
        *这一步是 Lasso 产生稀疏解的关键：当相关性 $\rho_j$ 小于正则化强度 $\alpha$ 时，权重会被直接置为 0。*
    *   **残差更新 (Residual Update)**:
        $$ r \leftarrow r - (w_j^{(new)} - w_j^{(old)}) x_j $$
        *实时维护残差向量，避免了 $O(N \cdot P)$ 的矩阵乘法，将复杂度降为 $O(N)$。*
3.  **收敛判断**: 记录本轮迭代中权重的最大变化量 `max_change`。若 `max_change < tol` 或达到 `max_iter`，则停止。
