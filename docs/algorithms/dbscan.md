# DBSCAN

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是基于密度的聚类算法，能够发现任意形状的簇，并可识别噪声点。它不需要预先指定簇的数量，对异常值鲁棒。

## 数学模型

DBSCAN 基于两个关键参数：邻域半径 $\epsilon$（eps）和最小点数 $minPts$。

核心概念包括：

$\epsilon$-邻域：样本 $x$ 的 $\epsilon$-邻域包含所有距离不超过 $\epsilon$ 的样本
$$ N_\epsilon(x) = \{ x_j \in D \mid \text{dist}(x, x_j) \leq \epsilon \} $$

核心点：$|N_\epsilon(x)| \geq minPts$ 的样本，即邻域内至少有 $minPts$ 个点。

边界点：不是核心点，但在某个核心点的 $\epsilon$-邻域内。

噪声点：既不是核心点也不是边界点的样本。

直接密度可达：若 $x_j \in N_\epsilon(x_i)$ 且 $x_i$ 是核心点，则 $x_j$ 从 $x_i$ 直接密度可达。

密度可达：存在样本序列 $p_1, \ldots, p_m$，其中 $p_1=x, p_m=y$，且 $p_{i+1}$ 从 $p_i$ 直接密度可达。

密度相连：存在样本 $o$，使得 $x$ 和 $y$ 都从 $o$ 密度可达。

簇定义为满足以下条件的非空集合 $C$：
1. 最大性：若 $x \in C$ 且 $y$ 从 $x$ 密度可达，则 $y \in C$
2. 连通性：$\forall x, y \in C$，$x$ 和 $y$ 密度相连

## 模型假设

DBSCAN 假设簇是密度相连的区域；噪声点分布在低密度区域；簇的密度相近（全局 $\epsilon$ 和 $minPts$ 适用）。对密度差异大的数据集效果不佳，高维数据因距离度量失效而性能下降。

## 模型特点

DBSCAN 的优点在于无需指定簇数、可发现任意形状的簇、对噪声和异常值鲁棒、对初始值不敏感（确定性算法）、可识别噪声点。其局限性是对参数 $\epsilon$ 和 $minPts$ 敏感、对密度差异大的簇效果差、高维数据效果不佳（维数灾难）、大规模数据计算开销大（需计算所有点对距离）。

## 参数选择

$\epsilon$ 和 $minPts$ 的选择对结果影响很大。

$minPts$ 的经验法则是至少为特征维度 $d$ 加 1（$minPts \geq d+1$），常用值为 4-10。较大的 $minPts$ 使簇更稠密，较小的值可能将噪声误判为簇。

$\epsilon$ 可用 K-距离图选择：对每个样本计算到第 $K$ 近邻的距离（$K=minPts$），将距离排序后绘图。选择曲线"肘部"对应的距离作为 $\epsilon$。该点之前是簇内距离，之后是簇间距离或噪声。

也可通过多次试验，选择使噪声比例合理（如 5%-15%）、簇结构清晰的参数组合。

## 算法流程

DBSCAN 的算法流程如下：

1. 初始化所有点为未访问
2. 遍历每个未访问的点 $p$：
   - 标记 $p$ 为已访问
   - 计算 $N_\epsilon(p)$
   - 若 $|N_\epsilon(p)| < minPts$，标记 $p$ 为噪声（可能后续被吸收为边界点）
   - 否则创建新簇 $C$，将 $p$ 加入 $C$
   - 对 $N_\epsilon(p)$ 中每个点 $q$：
     - 若 $q$ 未访问，标记为已访问并计算其邻域
     - 若 $|N_\epsilon(q)| \geq minPts$，将 $N_\epsilon(q)$ 加入 $N_\epsilon(p)$（扩展核心点）
     - 若 $q$ 不属于任何簇，将 $q$ 加入 $C$
3. 重复直到所有点被访问

算法时间复杂度为 $O(n^2)$，使用空间索引（KD 树、R 树）可降至 $O(n \log n)$。

## 变体

OPTICS（Ordering Points To Identify the Clustering Structure）放宽 $\epsilon$ 固定要求，生成可达距离图，可发现密度不同的簇，但计算开销更大。

HDBSCAN（Hierarchical DBSCAN）结合层次聚类和 DBSCAN，自动选择最优聚类，对参数不敏感，性能更稳健。

PDBSCAN（Parallel DBSCAN）利用空间划分和并行计算加速大规模数据处理。

Incremental DBSCAN 支持动态添加和删除数据点，适用于流数据场景。
