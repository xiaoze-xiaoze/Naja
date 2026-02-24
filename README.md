# Naja(ML Library for Rust)

Naja 是一个面向工程使用的 Rust 机器学习库，目标是在保持 Rust 风格与可维护性的前提下，为常见算法提供一致、可组合的 API。

## 目标与特性

- 统一的 Model / Fit / Predict-Transform 使用范式
- 最小核心层（core）与算法实现解耦，便于扩展
- 统一的数值计算与数据校验入口
- 组件与算法文档保持一致结构

## 快速开始

### 监督学习

```rust
let model = LinearRegression::new()
    .intercept(true)
    .penalty(Penalty::Ridge { alpha: 1e-2 });
let solution = model.fit(&x_train, &y_train)?;
let y_pred = solution.predict(&x_test)?;
```

### 非监督学习

```rust
let model = KMeans::new()
    .k(8)
    .max_iter(300);
let solution = model.fit(&x)?;
let labels = solution.predict(&x)?;
```

## API 设计（三阶段范式）

| 阶段 | 方法 | 职责 |
|:----:|:----:|:----:|
| Model | `Model::new().param(...)` | 创建模型并链式配置超参数 |
| Fit | `fit(&x, &y)` | 训练/求解 |
| Predict / Transform | `predict(&x)` / `transform(&x)` | 推理/变换 |

- Model 负责生成未拟合模型，支持链式配置
- Fit 负责训练并返回已拟合对象
- Predict / Transform 负责预测或数据变换

## 算法目录

1. Linear Regression
2. Logistic Regression
3. SVM
4. KNN
5. Naive Bayes
6. Decision Tree
7. Random Forest
8. XGBoost
9. KMeans
10. DBSCAN
11. Gaussian Mixture Model
12. LDA

## 项目结构

```
Naja/
├── Cargo.toml                         # crate 元信息与依赖
├── src/
│   ├── lib.rs                         # 模块声明与公开导出
│   ├── core/                          # 仅为算法服务的最小公共层
│   │   ├── mod.rs                     # core 入口；集中 re-export（给 algorithms 用）
│   │   ├── error.rs                   # 错误类型与 Result 定义
│   │   ├── traits.rs                  # fit/predict/transform 等最小 trait
│   │   ├── data/                      # Dataset + 基础数据校验工具
│   │   │   ├── mod.rs
│   │   │   ├── dataset.rs
│   │   │   └── validate.rs
│   │   └── compute/                   # ndarray/faer 的统一适配入口
│   │       ├── mod.rs
│   │       ├── types.rs
│   │       └── ops.rs
│   ├── preprocessing/
│   │   ├── mod.rs                     # 统一导出 + Pipeline 类型
│   │   ├── traits.rs                  # 预处理核心 trait (Estimator, Transformer)
│   │   ├── scaler/                    # 特征缩放
│   │   │   ├── mod.rs
│   │   │   ├── standard.rs            # StandardScaler
│   │   │   ├── minmax.rs              # MinMaxScaler
│   │   │   └── robust.rs              # RobustScaler
│   │   ├── encoder/                   # 数据编码
│   │   │   ├── mod.rs
│   │   │   ├── onehot.rs              # OneHotEncoder
│   │   │   └── label.rs               # LabelEncoder
│   │   ├── imputer/                   # 缺失值处理
│   │   │   ├── mod.rs
│   │   │   └── simple.rs              # SimpleImputer
│   │   └── pipeline/                  # 预处理流水线
│   │       ├── mod.rs
│   │       └── compose.rs
│   ├── io/                            # 模型导入导出 (ONNX 等)
│   │   ├── mod.rs
│   │   └── onnx.rs
│   ├── metrics/                       # 指标：分类/回归/聚类
│   │   ├── mod.rs                     # metrics 模块入口
│   │   ├── classifier.rs              # 分类指标
│   │   ├── regressor.rs               # 回归指标
│   │   └── clusterer.rs               # 聚类指标
│   └── algorithms/                    # 算法实现
│       ├── mod.rs
│       ├── linrg.rs
│       ├── logrg.rs
│       ├── svm.rs
│       ├── knn.rs
│       ├── nbayes.rs
│       ├── dtree.rs
│       ├── rndfst.rs
│       ├── xgb.rs
│       ├── kmeans.rs
│       ├── dbscan.rs
│       ├── gmm.rs
│       ├── pca.rs
│       └── lda.rs
└── examples/
    └── linrg_demo.rs
```

## 文档索引

### 组件文档

- **Core Traits**（核心抽象） -> [traits.md](docs/components/traits.md)
- **core::error**（错误与 Result） -> [error.md](docs/components/error.md)
- **core::compute**（数值运算） -> [compute.md](docs/components/compute.md)
- **core::data**（Dataset 与 validate） -> [data.md](docs/components/data.md)
- **preprocessing**（数据预处理） -> [preprocessing.md](docs/components/preprocessing.md)
- **Metrics 指标库** -> [metrics.md](docs/components/metrics.md)
- **Model Export (ONNX)** -> [onnx.md](docs/components/onnx.md)

### 算法文档

- 线性回归 -> [linrg.md](docs/algorithms/linrg.md)
