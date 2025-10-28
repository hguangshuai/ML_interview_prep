# 相对熵/交叉熵以及K-L散度

## English Interview Answer
Entropy measures uncertainty, H(P) = −Σ p log p. Cross‑Entropy H(P,Q) = −Σ p log q, and KL Divergence KL(P||Q) = Σ p log(p/q) = H(P,Q) − H(P). In classification, minimizing Cross‑Entropy is equivalent to minimizing KL(P||Q) and to maximizing likelihood.

## 中文知识点解释（含英文术语标注）
- 熵（Entropy）：不确定性度量；概率越均匀，熵越大。
- 交叉熵（Cross‑Entropy）：真实分布 P 与模型分布 Q 的距离度量 H(P,Q)；训练中最常用的分类损失。
- 相对熵/散度（KL Divergence, Kullback‑Leibler）：KL(P||Q) = H(P,Q) − H(P)，非对称（asymmetric），不是度量（metric）。
- 分类训练：最小化交叉熵 ⇔ 最小化 KL(P||Q) ⇔ 最大似然（Maximum Likelihood）。

## 核心概念

### 熵 (Entropy)

**信息论中的不确定性度量**

```python
H(X) = -Σ p(x) log p(x)
```

**直觉理解:** 一个事件的"惊喜程度"
- 低概率事件 = 高惊喜 = 高信息量
- 高概率事件 = 低惊喜 = 低信息量

### 交叉熵 (Cross-Entropy)

**衡量两个概率分布之间的距离**

```python
H(P, Q) = -Σ p(x) log q(x)
```

**机器学习中常见形式:**
```python
# 二分类
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

# 多分类
L = -Σ y_i * log(ŷ_i)
```

### K-L散度 (KL Divergence)

**衡量两个概率分布的差异**

```python
KL(P||Q) = Σ p(x) log(p(x)/q(x))
          = H(P, Q) - H(P)
```

**性质:**
- KL(P||Q) ≥ 0，当且仅当P=Q时为0
- **非对称性**: KL(P||Q) ≠ KL(Q||P)
- 不满足三角不等式（不是真正的距离）

## 直觉理解

### 1. 熵 (Entropy)

**不确定性度量**
```python
# 掷骰子
fair_die = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
entropy_fair = -sum([p * np.log2(p) for p in fair_die])  # ≈ 2.58

weighted_die = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
entropy_weighted = -sum([p * np.log2(p) for p in weighted_die])  # ≈ 1.86
```

### 2. 交叉熵 (Cross-Entropy)

**模型预测分布 vs 真实分布**

```python
# 真实标签
y_true = [1, 0, 0]  # one-hot encoding

# 模型预测
y_pred1 = [0.9, 0.05, 0.05]  # 好的预测
y_pred2 = [0.3, 0.4, 0.3]     # 差的预测

# 交叉熵
cross_entropy1 = -sum(y_true * np.log(y_pred1))  # 小
cross_entropy2 = -sum(y_true * np.log(y_pred2))  # 大
```

### 3. K-L散度

**信息增益，减少的不确定性**

```python
# 真实分布
P = [0.5, 0.3, 0.2]

# 模型分布
Q = [0.6, 0.2, 0.2]

# KL散度
kl = sum([P[i] * np.log(P[i]/Q[i]) for i in range(len(P))])
```

## 在机器学习中的应用

### 分类问题的Cross-Entropy

```python
# Softmax + Cross-Entropy
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10))

# 示例
logits = np.array([2.0, 1.0, 0.1])
y_true = np.array([1, 0, 0])

probs = softmax(logits)
loss = cross_entropy(y_true, probs)
```

### 为什么使用Cross-Entropy？

1. **最小化KL散度**: 等价于最小化KL(P||Q)
2. **梯度友好**: 梯度计算简单
3. **数值稳定**: 配合softmax使用
4. **理论支撑**: 最大似然估计

### 从KL散度推导Cross-Entropy

**训练目标: 最小化KL散度**
```python
min KL(P||Q) = min [H(P, Q) - H(P)]
            = min H(P, Q)  # H(P)是常数
```

**因此最小化交叉熵 ≡ 最小化KL散度！**

## 代码实现

```python
import numpy as np

def entropy(p):
    """计算熵"""
    p = np.array(p)
    return -np.sum(p * np.log(p + 1e-10))

def cross_entropy(p, q):
    """计算交叉熵"""
    p = np.array(p)
    q = np.array(q)
    return -np.sum(p * np.log(q + 1e-10))

def kl_divergence(p, q):
    """计算KL散度"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

# 示例
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.6, 0.2, 0.2])

print(f"Entropy of P: {entropy(P):.4f}")
print(f"Cross-Entropy H(P,Q): {cross_entropy(P, Q):.4f}")
print(f"KL Divergence: {kl_divergence(P, Q):.4f}")
print(f"Verification: {cross_entropy(P, Q) - entropy(P):.4f}")
```

## 关系总结

```
交叉熵 = KL散度 + 熵

H(P, Q) = KL(P||Q) + H(P)
```

**训练模型时:**
- 最小化交叉熵
- = 最小化KL散度
- = 使模型分布Q接近真实分布P

## 面试要点

1. **熵**: 不确定性度量 H(P) = -Σ p log p
2. **交叉熵**: 预测分布与真实分布的距离 H(P,Q) = -Σ p log q
3. **KL散度**: 两个分布的差异 KL(P||Q) = H(P,Q) - H(P)
4. **最小化交叉熵 ≡ 最小化KL散度**
5. **不对称性**: KL(P||Q) ≠ KL(Q||P)
