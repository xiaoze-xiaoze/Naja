# core::data（Dataset 与 validate）

## 概述

`core::data` 提供轻量数据容器 `Dataset` 与配套校验函数，支持零拷贝视图，适合算法示例与中小规模数据场景。

## 模块位置

- `src/core/data/dataset.rs`
- `src/core/data/validate.rs`

---

## 类型定义

### Dataset

```rust
pub struct Dataset {
    pub records: Matrix,
    pub targets: Vector,
    pub feature_names: Vec<String>,
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `records` | `Matrix` | 特征矩阵（n_samples × n_features） |
| `targets` | `Vector` | 目标向量（n_samples） |
| `feature_names` | `Vec<String>` | 特征名称列表 |

### DatasetView<'a>

```rust
pub struct DatasetView<'a> {
    pub records: MatrixView<'a>,
    pub targets: VectorView<'a>,
    pub feature_names: &'a [String],
}
```

`Dataset` 的零拷贝视图，用于避免数据复制。

---

## Dataset 方法

| 方法 | 签名 | 含义 |
|------|------|------|
| `new` | `fn new(records: Matrix, targets: Vector) -> Result<Self>` | 构造数据集并校验维度一致性 |
| `with_feature_names` | `fn with_feature_names(self, names: Vec<String>) -> Result<Self>` | 设置特征名（校验数量一致） |
| `as_view` | `fn as_view(&self) -> DatasetView<'_>` | 获取零拷贝视图 |
| `split` | `fn split(&self, test_ratio: f64) -> Result<(DatasetView<'_>, DatasetView<'_>)>` | 零拷贝划分训练/测试集 |
| `nrows` | `fn nrows(&self) -> usize` | 返回样本数 |
| `ncols` | `fn ncols(&self) -> usize` | 返回特征数 |

---

## DatasetView 方法

| 方法 | 签名 | 含义 |
|------|------|------|
| `to_owned` | `fn to_owned(&self) -> Dataset` | 转换为拥有所有权的 Dataset |
| `nrows` | `fn nrows(&self) -> usize` | 返回样本数 |
| `ncols` | `fn ncols(&self) -> usize` | 返回特征数 |

---

## 校验函数

| 函数 | 签名 | 含义 | 错误类型 |
|------|------|------|----------|
| `check_dimensions` | `fn(records: MatrixView<'_>, targets: VectorView<'_>) -> Result<()>` | 校验样本数一致 | `InvalidShape` |
| `check_feature` | `fn(n_features: usize, names: &[String]) -> Result<()>` | 校验特征数与特征名数量一致 | `InvalidShape` |
| `check_split_ratio` | `fn(ratio: f64) -> Result<()>` | 校验 split 比例在 [0,1] 范围 | `InvalidParam` |

---

## 使用示例

### 基本使用

```rust
use naja::core::data::{Dataset, DatasetView};

let dataset = Dataset::new(records, targets)?;
let dataset = dataset.with_feature_names(vec!["age".into(), "income".into()])?;

// 零拷贝 split
let (train, test) = dataset.split(0.2)?;

// 直接传给算法
let model = LinearRegression::new();
let fitted = model.fit_supervised(train.records, train.targets)?;
```

### 视图转换

```rust
let view = dataset.as_view();

// 需要独立所有权时
let owned = view.to_owned();
```

---

## 适用场景

- 快速原型与示例代码
- 中小规模、内存可容纳的数据
- 不需要复杂流水线的训练流程
- 需要零拷贝的高效数据切分

---

## 注意事项

- `split` 按顺序切分，**不进行随机打乱**；如需随机打乱请在传入前处理
- `DatasetView` 生命周期绑定到原 `Dataset`，不可超出原数据存活范围
- `Dataset` 不实现 `Clone`，避免意外深拷贝；如需复制请用 `to_owned()`
