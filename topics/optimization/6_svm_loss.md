# SVM的Loss是什么

## 核心答案

**SVM使用Hinge Loss，目标是最大化分类间隔**

## 详细分析

### 1. SVM优化目标

**软间隔SVM的完整优化问题:**
```python
min (1/2)||w||² + C Σ ξ_i
subject to:
  y_i(w·x_i + b) ≥ 1 - ξ_i
  ξ_i ≥ 0
```

**等价的无约束形式:**
```python
min Σ L_hinge(y_i * f(x_i)) + λ||w||²
```

其中：
- L_hinge = max(0, 1 - y_i * f(x_i))  # Hinge Loss
- λ = 1/(2C)  # 正则化系数
- f(x_i) = w·x_i + b  # 决策函数

### 2. Hinge Loss定义

```python
L_hinge(y, f(x)) = max(0, 1 - y * f(x))
```

**分段形式:**
```python
L_hinge = {
    0,           if y·f(x) ≥ 1      # 正确且远离边界
    1 - y·f(x),  if y·f(x) < 1       # 错误或接近边界
}
```

**直觉理解:**
- 当**y·f(x) ≥ 1**时，分类器置信度高，loss=0
- 当**y·f(x) < 1**时，需要惩罚，惩罚量为1-y·f(x)

### 3. 与其他Loss对比

#### Hinge Loss vs Cross-Entropy

**Hinge Loss:**
```python
L = max(0, 1 - y·f(x))
```

- 当正确分类且远离边界时，loss=0
- 关注**分类正确性**，不是概率校准
- 稀疏性：只关注边界附近的样本（支持向量）

**Cross-Entropy:**
```python
L = -log(σ(y·f(x)))
```

- 总是提供非零梯度
- 关注**概率校准**
- 每个样本都贡献梯度

### 4. 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt

def hinge_loss(y_true, y_pred):
    """Hinge Loss"""
    return np.maximum(0, 1 - y_true * y_pred)

def cross_entropy_loss(y_true, y_pred_prob):
    """Cross-Entropy Loss"""
    return -np.log(y_pred_prob + 1e-10)

def zero_one_loss(y_true, y_pred):
    """0-1 Loss"""
    return (y_true != y_pred).astype(float)

# 可视化不同Loss函数
y_true = 1
y_fx = np.linspace(-2, 2, 100)

hinge_vals = [hinge_loss(y_true, fx) for fx in y_fx]
ce_vals = [-np.log(1 / (1 + np.exp(-fx))) for fx in y_fx]
zero_one_vals = [zero_one_loss(y_true, 1 if fx > 0 else -1) for fx in y_fx]

plt.figure(figsize=(10, 6))
plt.plot(y_fx, hinge_vals, label='Hinge Loss', linewidth=2)
plt.plot(y_fx, ce_vals, label='Cross-Entropy', linewidth=2)
plt.plot(y_fx, zero_one_vals, label='0-1 Loss', linewidth=2, linestyle='--')
plt.axvline(x=1, color='r', linestyle=':', label='Margin Boundary')
plt.xlabel('y * f(x)')
plt.ylabel('Loss')
plt.title('Loss Function Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5. SVM为什么要用Hinge Loss？

#### 数学优势

1. **凸优化**: Hinge Loss是凸函数，保证全局最优
2. **稀疏性**: 只关注边界附近的样本
3. **最大间隔原理**: 最大化分类间隔

#### 几何直觉

```
正确分类区域: y·f(x) ≥ 1  → Loss = 0
间隔内部区域: 0 < y·f(x) < 1  → Loss = 1 - y·f(x)
错误分类: y·f(x) ≤ 0  → Loss = 1 - y·f(x)
```

**只有支持向量（边界上的点）才影响模型！**

### 6. 完整的SVM训练

```python
from sklearn.svm import SVC
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.sign(X[:, 0] + X[:, 1])

# SVM模型（本质上最小化Hinge Loss）
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# 支持向量
print(f"Number of support vectors: {model.n_support_}")
print(f"Support vectors: {model.support_vectors_}")

# 决策边界
w = model.coef_[0]
b = model.intercept_[0]
print(f"Weights: {w}")
print(f"Bias: {b}")
```

### 7. 梯度（对于L2正则化）

```python
def svm_loss_and_gradient(X, y, w, C):
    """
    计算SVM的Hinge Loss和梯度
    """
    scores = X @ w
    margins = y * scores
    
    # Hinge Loss
    loss = np.sum(np.maximum(0, 1 - margins)) / len(y)
    loss += 0.5 / C * np.dot(w, w)  # L2正则化
    
    # 梯度
    gradient = np.zeros_like(w)
    for i in range(len(y)):
        if margins[i] < 1:
            gradient -= y[i] * X[i, :]
    gradient /= len(y)
    gradient += w / C
    
    return loss, gradient

# 训练SVM
def train_svm(X, y, learning_rate=0.01, epochs=1000, C=1.0):
    w = np.random.randn(X.shape[1])
    
    for epoch in range(epochs):
        loss, grad = svm_loss_and_gradient(X, y, w, C)
        w -= learning_rate * grad
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w
```

### 8. 软间隔vs硬间隔

**硬间隔（ξ_i=0）:**
```python
min (1/2)||w||²
subject to: y_i(w·x_i + b) ≥ 1
```
- 要求所有点都正确分类
- 对噪声敏感

**软间隔（允许ξ_i>0）:**
```python
min (1/2)||w||² + C Σ ξ_i
subject to: y_i(w·x_i + b) ≥ 1 - ξ_i
```
- 允许一些点落在间隔内或错误分类
- 通过C参数控制惩罚强度

### 9. 关键特点总结

| 特性 | Hinge Loss | Cross-Entropy |
|------|-----------|---------------|
| 零点特性 | 有零点（当y·f(x)≥1） | 无零点 |
| 概率校准 | 不考虑 | 考虑 |
| 稀疏性 | 高（支持向量） | 低（所有样本） |
| 梯度 | 分段常数 | 平滑 |
| 适用性 | 分类正确性 | 概率估计 |

### 面试要点

1. **SVM使用Hinge Loss**: L = max(0, 1 - y·f(x))
2. **目标**: 最小化Hinge Loss + 正则化项
3. **稀疏性**: 只关注支持向量
4. **与Cross-Entropy的区别**: Hinge Loss有零点，不关注概率
5. **最大间隔原理**: 最大化分类间隔
