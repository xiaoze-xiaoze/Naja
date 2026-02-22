# core::data（Dataset 与 validate）

## 概述

`core::data` 提供轻量数据容器 `Dataset` 与配套校验函数，支持零拷贝视图，适合算法示例与中小规模数据场景。

## 模块位置

- `src/core/data/dataset.rs`
- `src/core/data/validate.rs`

## 类型定义

### Dataset

- **字段**:
    - `records: Matrix`
    - `targets: Vector`
    - `feature_names: Vec<String>`
- **含义**: 统一管理特征矩阵、目标向量与特征名
- **用途**: 为算法与示例提供一致的数据入口

### DatasetView<'a>

- **字段**:
    - `records: MatrixView<'a>`
    - `targets: VectorView<'a>`
    - `feature_names: &'a [String]`
- **含义**: Dataset 的零拷贝视图
- **用途**: 避免 split 时数据拷贝，直接传递给算法

## Dataset 方法

- **Dataset::new**
    - **签名**: `fn new(records: Matrix, targets: Vector) -> Result<Self>`
    - **含义**: 构造数据集并校验维度
    - **用途**: 标准数据集入口

- **Dataset::with_feature_names**
    - **签名**: `fn with_feature_names(self, names: Vec<String>) -> Result<Self>`
    - **含义**: 设置特征名并校验数量
    - **用途**: 提升可读性与可解释性

- **Dataset::as_view**
    - **签名**: `fn as_view(&self) -> DatasetView<'_>`
    - **含义**: 获取数据集的零拷贝视图
    - **用途**: 传递给接受 View 的函数

- **Dataset::split**
    - **签名**: `fn split(&self, test_ratio: f64) -> Result<(DatasetView<'_>, DatasetView<'_>)>`
    - **含义**: 零拷贝划分训练集/测试集
    - **用途**: 快速切分数据，无内存开销
    - **注意**: `test_ratio` 必须在 `[0.0, 1.0]` 范围内

## DatasetView 方法

- **DatasetView::to_owned**
    - **签名**: `fn to_owned(&self) -> Dataset`
    - **含义**: 将视图转换为 owned Dataset
    - **用途**: 需要独立所有权时使用

- **DatasetView::nrows / ncols**
    - **签名**: `fn nrows(&self) -> usize`, `fn ncols(&self) -> usize`
    - **含义**: 获取维度信息

## 校验函数

- **check_dimensions**
    - **签名**: `fn(records: MatrixView<'_>, targets: VectorView<'_>) -> Result<()>`
    - **含义**: 校验样本数一致性
    - **错误**: `InvalidShape`

- **check_feature**
    - **签名**: `fn(n_features: usize, names: &[String]) -> Result<()>`
    - **含义**: 校验特征数与特征名数量一致性
    - **错误**: `InvalidShape`

## 使用示例

```rust
use naja::core::data::{Dataset, DatasetView};

let dataset = Dataset::new(records, targets)?;
let dataset = dataset.with_feature_names(vec!["age".into(), "income".into()])?;

// 零拷贝 split
let (train, test) = dataset.split(0.2)?;

// 直接传给算法（traits 接受 View）
let model = LinearRegression::new().fit(train.records, train.targets)?;

// 如需独立所有权
let owned_train = train.to_owned();
```

## 适用场景

- 快速原型与示例
- 中小规模、内存可容纳的数据
- 不需要复杂流水线的训练流程
- 需要零拷贝的高效数据切分

## 注意事项

- `split` 按顺序切分，不进行随机打乱
- `DatasetView` 生命周期绑定到原 `Dataset`
- `Dataset` 不再实现 `Clone`，避免意外深拷贝
