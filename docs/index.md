# Naja 文档

## 概述

Naja 是一个面向工程使用的 Rust 机器学习库，提供一致的 Model / Fit / Predict-Transform 使用范式。

## 快速入门

→ [快速入门指南](guide/quickstart.md)

## 模块文档

### 核心模块 (core)

| 模块 | 说明 |
|------|------|
| [traits](core/traits.md) | 核心抽象 — fit/predict/transform trait |
| [error](core/error.md) | 错误处理 — 统一 Result 与 Error 类型 |
| [compute](core/compute.md) | 数值运算 — ndarray/faer 封装 |
| [data](core/data.md) | 数据容器 — Dataset 与校验函数 |

### 预处理与流水线

| 模块 | 说明 |
|------|------|
| [preprocessing](preprocessing/index.md) | 数据预处理 — StandardScaler / MinMaxScaler / RobustScaler |
| [pipeline](pipeline/index.md) | 流水线 — 预处理与模型的组合 |

### 评估与导出

| 模块 | 说明 |
|------|------|
| [metrics](metrics/index.md) | 指标库 — 回归/分类/聚类评估指标 |
| [io](io/index.md) | 模型导出 — ONNX（规划中） |

### 算法

→ [算法文档总览](algorithms/index.md)

## API 设计范式

| 阶段 | 方法 | 职责 |
|:----:|:----:|:----:|
| Model | `Model::new().param(...)` | 创建模型并链式配置超参数 |
| Fit | `fit(&x, &y)` / `fit(&x)` | 训练/求解 |
| Predict / Transform | `predict(&x)` / `transform(&x)` | 推理/变换 |

## 文件结构

```
docs/
├── index.md              # 本文档
├── guide/
│   └── quickstart.md     # 快速入门
├── core/                 # 核心模块
│   ├── traits.md
│   ├── error.md
│   ├── compute.md
│   └── data.md
├── preprocessing/        # 预处理
│   └── index.md
├── pipeline/             # 流水线
│   └── index.md
├── metrics/              # 指标
│   └── index.md
├── io/                   # 输入输出
│   └── index.md
└── algorithms/           # 算法
    └── *.md
```
