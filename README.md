# Naja(ML Library for Rust)

Naja 是一个面向工程使用的 Rust 机器学习库，目标是在保持 Rust 风格与可维护性的前提下，为常见算法提供一致、可组合的 API。

## 目标与特性

- 统一的 Model / Fit / Predict-Transform 使用范式
- 最小核心层（core）与算法实现解耦，便于扩展
- 统一的数值计算与数据校验入口
- 组件与算法文档保持一致结构

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
8. Gradient Boosting Machine
1. KMeans
2.  DBSCAN
3.  Gaussian Mixture Model
4.  LDA

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
│   ├── preprocessing/                 # 预处理模块
│   │   ├── mod.rs                     # 统一导出 + Pipeline 类型
│   │   ├── scaler/                    # 特征缩放
│   │   │   ├── mod.rs
│   │   │   ├── standard.rs            # StandardScaler
│   │   │   ├── minmax.rs              # MinMaxScaler
│   │   │   └── robust.rs              # RobustScaler
│   │   ├── encoder/                   # 数据编码
│   │   │   ├── mod.rs
│   │   │   ├── onehot.rs              # OneHotEncoder
│   │   │   └── label.rs               # LabelEncoder
│   │   └── imputer/                   # 缺失值处理
│   │       ├── mod.rs
│   │       └── simple.rs              # SimpleImputer
│   │── pipeline/                      # 流水线
│   │   ├── mod.rs
│   │   └── compose.rs
│   ├── io/                            # 模型导入导出
│   │   ├── mod.rs
│   │   └── onnx.rs
│   ├── metrics/                       # 指标
│   │   ├── mod.rs                     # metrics 模块入口
│   │   ├── classifier.rs              # 分类指标
│   │   ├── regressor.rs               # 回归指标
│   │   └── clusterer.rs               # 聚类指标
│   └── algorithms/                    # 算法实现
│       ├── mod.rs
│       ├── linrg/
│       ├── logrg/
│       ├── svm/
│       ├── knn/
│       ├── nbayes/
│       ├── dtree/
│       ├── rndfst/
│       ├── gbm/
│       ├── kmeans/
│       ├── dbscan/
│       ├── gmm/
│       ├── pca/
│       └── lda/
├── tests/
└── examples/
```

## 文档索引

### 快速入门

- [快速入门指南](docs/guide/quickstart.md) — 端到端示例

### 核心模块 (core)

- [core 概览](docs/core/index.md)
- [traits](docs/core/traits.md) — 核心抽象（fit/predict/transform）
- [error](docs/core/error.md) — 错误处理
- [compute](docs/core/compute.md) — 数值运算
- [data](docs/core/data.md) — 数据容器

### 预处理与流水线

- [preprocessing](docs/preprocessing/index.md) — 数据预处理（StandardScaler / MinMaxScaler / RobustScaler）
- [pipeline](docs/pipeline/index.md) — 流水线

### 评估与导出

- [metrics](docs/metrics/index.md) — 指标库
- [io](docs/io/index.md) — 模型导出（ONNX）

### 算法

- [算法概览](docs/algorithms/index.md)
- [线性回归](docs/algorithms/linrg.md)
