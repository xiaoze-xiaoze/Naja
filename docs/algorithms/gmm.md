# Gaussian Mixture Model

高斯混合模型（GMM）是基于概率密度的聚类方法，假设数据由多个高斯分布混合生成。它通过 EM 算法估计各高斯分量的参数，是软聚类的经典方法。

## 数学模型

GMM 假设数据服从 $K$ 个高斯分布的混合：

$$ p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) $$

其中 $\pi_k$ 是混合系数（$\pi_k > 0, \sum_k \pi_k = 1$），$\mathcal{N}(x | \mu_k, \Sigma_k)$ 是第 $k$ 个高斯分量：

$$ \mathcal{N}(x | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2}(x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k) \right) $$

引入隐变量 $z_i \in \{1,\ldots,K\}$ 表示样本 $x_i$ 的生成簇，$P(z_i=k)=\pi_k$。完全数据的对数似然为

$$ \log p(X, Z) = \sum_{i=1}^n \sum_{k=1}^K \mathbb{I}(z_i=k) \left[ \log \pi_k + \log \mathcal{N}(x_i | \mu_k, \Sigma_k) \right] $$

由于 $Z$ 未知，采用 EM 算法最大化期望对数似然。

## EM 算法

EM 算法交替执行 E 步和 M 步：

**E 步**：计算后验概率（责任度）

$$ \gamma_{ik} = P(z_i=k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)} $$

$\gamma_{ik}$ 表示样本 $i$ 由分量 $k$ 生成的概率。

**M 步**：最大化期望对数似然，更新参数

$$ \mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i $$

$$ \Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T $$

$$ \pi_k^{\text{new}} = \frac{N_k}{n} $$

其中 $N_k = \sum_i \gamma_{ik}$ 是分量 $k$ 的有效样本数。

重复直到对数似然收敛：

$$ \log p(X) = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) \right) $$

## 模型假设

GMM 假设数据由多个高斯分布混合生成；各簇呈椭球形分布；样本独立同分布。对非高斯分布或复杂流形结构的数据拟合能力有限。

## 模型特点

GMM 的优点在于提供软聚类（概率形式的不确定性）、可估计概率密度、有坚实的概率论基础、可处理不同大小和形状的簇（通过协方差矩阵）。其局限性是对初始值敏感、可能收敛到局部最优、需指定 $K$、高维数据协方差矩阵估计不稳定、计算开销较大（需矩阵求逆）。

## 协方差类型

协方差矩阵的结构影响模型复杂度和拟合能力。

完整协方差（full）每个分量有独立的完整协方差矩阵 $\Sigma_k$，参数最多、拟合能力最强，但易过拟合且计算成本高。

共享协方差（tied）所有分量共享同一协方差矩阵 $\Sigma$，参数较少，适合样本量少的场景。

对角协方差（diag）协方差矩阵只有对角元素非零，假设特征条件独立，参数数量为 $O(Kd)$。

球形协方差（spherical）协方差矩阵为 $\sigma_k^2 I$，假设各向同性且特征独立，参数最少。

选择取决于数据量、特征维度和问题需求，常用对角或球形协方差平衡复杂度与拟合能力。

## K 值选择

选择 $K$ 的常用方法包括：

贝叶斯信息准则（BIC）

$$ BIC = -2 \log L + p \log n $$

其中 $L$ 是似然值，$p$ 是参数数量，$n$ 是样本数。选择 BIC 最小的 $K$。

赤池信息准则（AIC）

$$ AIC = -2 \log L + 2p $$

AIC 对复杂模型惩罚较轻，倾向于选择更大的 $K$。

轮廓系数或 Calinski-Harabasz 指数等聚类评估指标也可用于选择 $K$。

## 初始化策略

GMM 对初始值敏感，常用初始化方法包括：

K-Means 初始化：先用 K-Means 聚类，用其结果初始化 $\mu_k$、$\Sigma_k$ 和 $\pi_k$。稳定且通常能得到较好结果。

随机初始化：随机选择 $K$ 个样本作为初始均值，用全局协方差或单位矩阵初始化协方差，混合系数设为均匀分布。需多次运行取最优。

层次聚类初始化：用层次聚类得到初始划分，稳定性好但计算开销大。

为防止奇异矩阵（行列式为零），可对协方差矩阵加入小的正则项 $\Sigma_k + \epsilon I$。
