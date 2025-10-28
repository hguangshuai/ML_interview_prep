# MSE做Loss的LogisticRegression是Convex Problem吗？

## 答案

**不是。**用MSE做loss的LogisticRegression是**non-convex problem**。

## 详细分析

### 原因

虽然LogisticRegression本身是线性模型，但使用MSE损失函数结合sigmoid激活函数后，整个损失函数不再是convex的。

#### Convex函数定义

对于函数 f(x)，如果对于任意 x₁, x₂ 和 λ ∈ [0,1]：
```
f(λx₁ + (1-λ)x₂) ≤ λf(x₁) + (1-λ)f(x₂)
```

#### MSE Loss在LogisticRegression中

**模型:**
```python
h = σ(W·X + b) = 1 / (1 + e^(-z))
```

**MSE Loss:**
```python
L_MSE = (1/n) * Σ(y - σ(W·X + b))²
```

**关键问题:** Hessian矩阵的符号

对MSE损失函数求二阶导数，Hessian矩阵：
```python
∂²L_MSE/∂w_j∂w_k = (2/n) * Σ[h * (1-h) * x_j * x_k * (1 - 2h + y(2h - 1))]
```

由于存在 `(1 - 2h + y(2h - 1))` 这一项，根据 y 和 h 的值，Hessian矩阵可能不是半正定的，因此损失函数是**non-convex**的。

### 对比：Cross-Entropy Loss是Convex的

**Cross-Entropy Loss:**
```python
L_CE = -(1/n) * Σ[y * log(h) + (1-y) * log(1-h)]
```

**Hessian矩阵:**
```python
H_CE = (1/n) * Σ[h * (1-h) * x_j * x_k]
```

由于 h(1-h) > 0（sigmoid函数的性质），Hessian矩阵半正定，因此函数是convex的。

### 实际影响

**Non-Convex问题的挑战:**
- 可能存在**多个局部最小值**
- 优化可能陷入局部最优
- 难以保证收敛到全局最优
- 训练不稳定

### 代码示例

```python
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, random_state=42)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# MSE Loss
def mse_loss(w):
    h = sigmoid(X @ w)
    return np.mean((y - h)**2)

# Cross-Entropy Loss  
def crossentropy_loss(w):
    h = sigmoid(X @ w)
    return -np.mean(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))

# Check convexity by analyzing Hessian
def compute_hessian(loss_func, w):
    eps = 1e-5
    n = len(w)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            
            hessian[i, j] = (loss_func(w + eps*e_i + eps*e_j) 
                           - loss_func(w + eps*e_i) 
                           - loss_func(w + eps*e_j) 
                           + loss_func(w)) / (eps**2)
    return hessian

initial_w = np.random.randn(2)
mse_hessian = compute_hessian(mse_loss, initial_w)
mse_eigenvalues = np.linalg.eigvals(mse_hessian)
print(f"MSE is convex: {all(mse_eigenvalues >= -1e-10)}")  # False

ce_hessian = compute_hessian(crossentropy_loss, initial_w)
ce_eigenvalues = np.linalg.eigvals(ce_hessian)
print(f"Cross-Entropy is convex: {all(ce_eigenvalues >= -1e-10)}")  # True
```

### 总结

| 特性 | MSE Loss | Cross-Entropy Loss |
|------|----------|-------------------|
| Convex性 | ❌ Non-convex | ✅ Convex |
| 全局最优 | ❌ 不保证 | ✅ 保证 |
| 训练稳定性 | ❌ 不稳定 | ✅ 稳定 |
| 适用性 | ❌ 不适合分类 | ✅ 适合分类 |

### 面试要点

1. **MSE + LogisticRegression = Non-convex problem**
2. **Cross-Entropy + LogisticRegression = Convex problem**
3. **推荐在LogisticRegression中使用Cross-Entropy loss**
4. **理解Hessian矩阵在判断convexity中的作用**
