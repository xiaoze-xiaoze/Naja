# 算法文档

## 概述

Naja 提供常用的监督学习与无监督学习算法，所有算法遵循统一的 Model / Fit / Predict 范式。

## 算法总览

### 监督学习

#### 回归

| 算法 | 文档 | 说明 |
|------|------|------|
| Linear Regression | [linrg.md](linrg.md) | 线性回归（支持 L1/L2 正则化） |

#### 分类

| 算法 | 文档 | 说明 |
|------|------|------|
| Logistic Regression | [logrg.md](logrg.md) | 逻辑回归 |
| SVM | [svm.md](svm.md) | 支持向量机 |
| KNN | [knn.md](knn.md) | K 近邻 |
| Naive Bayes | [nbayes.md](nbayes.md) | 朴素贝叶斯 |
| Decision Tree | [dtree.md](dtree.md) | 决策树 |
| Random Forest | [rndfst.md](rndfst.md) | 随机森林 |
| XGBoost | [xgb.md](xgb.md) | 梯度提升 |

### 无监督学习

#### 聚类

| 算法 | 文档 | 说明 |
|------|------|------|
| KMeans | [kmeans.md](kmeans.md) | K 均值聚类 |
| DBSCAN | [dbscan.md](dbscan.md) | 基于密度的聚类 |
| GMM | [gmm.md](gmm.md) | 高斯混合模型 |

#### 降维

| 算法 | 文档 | 说明 |
|------|------|------|
| PCA | [pca.md](pca.md) | 主成分分析 |
| LDA | [lda.md](lda.md) | 线性判别分析 |

---

## 使用范式

所有算法遵循统一的三阶段范式：

```rust
// 1. 创建模型（链式配置超参数）
let model = Algorithm::new()
    .param1(value1)
    .param2(value2);

// 2. 训练
let fitted = model.fit_supervised(x, y)?;  // 监督学习
// 或
let fitted = model.fit_unsupervised(x)?;   // 无监督学习

// 3. 预测/变换
let result = fitted.predict(x)?;           // 预测
// 或
let result = fitted.transform(x)?;         // 变换（降维等）
```

---

## 按任务选择算法

### 回归任务

```
数据规模小 → Linear Regression
需要特征选择 → Linear Regression + L1 正则化
需要正则化 → Linear Regression + L2 正则化
```

### 分类任务

```
线性可分 → Logistic Regression / SVM (linear)
非线性 → SVM (RBF) / Random Forest / XGBoost
需要概率输出 → Logistic Regression
高维稀疏数据 → Naive Bayes
可解释性要求 → Decision Tree
```

### 聚类任务

```
簇形状球形、数量已知 → KMeans
簇形状任意、数量未知 → DBSCAN
软聚类、概率归属 → GMM
```

### 降维任务

```
无监督降维 → PCA
有监督降维（保留类别信息） → LDA
```

---

## 源码位置

所有算法实现位于 `src/algorithms/` 目录：

```
src/algorithms/
├── mod.rs
├── linrg.rs      # 线性回归
├── logrg.rs      # 逻辑回归
├── svm.rs
├── knn.rs
├── nbayes.rs
├── dtree.rs
├── rndfst.rs
├── xgb.rs
├── kmeans.rs
├── dbscan.rs
├── gmm.rs
├── pca.rs
└── lda.rs
```
