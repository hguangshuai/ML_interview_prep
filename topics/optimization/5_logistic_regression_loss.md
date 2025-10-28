# LogisticRegression的Loss推导

## English Interview Answer
Binary logistic regression uses the Cross‑Entropy (negative log‑likelihood) loss derived from a Bernoulli model with a sigmoid link. The gradient simplifies to ∇L = Xᵀ(ŷ − y). Cross‑Entropy avoids the vanishing‑gradient issue that arises when using MSE with a sigmoid.

## 中文知识点解释（含英文术语标注）
- 损失：交叉熵（Cross‑Entropy, Negative Log‑Likelihood），源自伯努利（Bernoulli）似然与 Sigmoid 链接（link）。
- 梯度：∇L = Xᵀ(ŷ − y)，形式简洁、数值稳定。
- 对比：MSE + Sigmoid 易出现梯度消失（vanishing gradient），优化缓慢；交叉熵收敛更快、凸性更好。

## 核心答案

**LogisticRegression使用Cross-Entropy Loss（二元交叉熵）**

## 完整推导

### 1. 模型设定

**Logistic函数（Sigmoid）:**
```python
h(x) = 1 / (1 + e^(-z))
     = 1 / (1 + e^(-W·X + b))
```

其中：
- z = W·X + b (线性组合)
- h(x) ∈ (0, 1) (概率)

### 2. 概率解释

**假设Y服从Bernoulli分布:**
```python
P(Y=1|X) = h(x) = p
P(Y=0|X) = 1 - h(x) = 1 - p
```

**联合概率:**
```python
P(Y=y|X) = p^y * (1-p)^(1-y)
```

### 3. 最大似然估计

**似然函数:**
```python
L(W) = Π P(y_i | x_i, W)
     = Π [h(x_i)]^(y_i) * [1-h(x_i)]^(1-y_i)
```

**对数似然:**
```python
l(W) = Σ [y_i * log(h(x_i)) + (1-y_i) * log(1-h(x_i))]
```

**最大化似然 ≡ 最小化负对数似然:**
```python
L(W) = -l(W) 
     = -Σ [y_i * log(h(x_i)) + (1-y_i) * log(1-h(x_i))]
     = Σ Cross-Entropy(y_i, h(x_i))
```

**这就是Cross-Entropy Loss！**

### 4. Cross-Entropy Loss公式

```python
L = -(1/n) * Σ [y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
```

**二元分类扩展:**
```python
# 当y=1时
L = -log(ŷ)

# 当y=0时  
L = -log(1-ŷ)
```

### 5. 梯度推导

**对权重W求导:**
```python
∂L/∂w_j = -(1/n) * Σ [y_i * (1/ŷ_i) * (∂ŷ_i/∂z_i) * x_j 
                     + (1-y_i) * (1/(1-ŷ_i)) * (-∂ŷ_i/∂z_i) * x_j]
```

**关键:** ∂ŷ/∂z = ŷ(1-ŷ) (sigmoid的导数)

**化简后:**
```python
∂L/∂w_j = -(1/n) * Σ [y_i * (1/ŷ_i) * ŷ_i(1-ŷ_i) * x_j 
                     - (1-y_i) * (1/(1-ŷ_i)) * ŷ_i(1-ŷ_i) * x_j]
        = -(1/n) * Σ [y_i * (1-ŷ_i) * x_j - (1-y_i) * ŷ_i * x_j]
        = -(1/n) * Σ [(y_i - ŷ_i) * x_j]
        = (1/n) * Σ [(ŷ_i - y_i) * x_j]
```

**最终梯度:**
```python
∇L = (1/n) * Xᵀ(ŷ - y)
```

### 6. 为什么不能用MSE？

MSE的梯度：
```python
∂L_MSE/∂w = -(2/n) * Σ (y - ŷ) * ŷ(1-ŷ) * x
```

当预测远离真实值时（ŷ≈0, y=1 或 ŷ≈1, y=0），梯度趋近0，学习缓慢！

Cross-Entropy的梯度：
```python
∂L_CE/∂w = -(1/n) * Σ (y - ŷ) * x
```

梯度与误差成正比，学习更有效！

### 代码实现

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Cross-Entropy Loss"""
        return -np.mean(y_true * np.log(y_pred + 1e-10) + 
                       (1 - y_true) * np.log(1 - y_pred + 1e-10))
    
    def gradient(self, X, y, y_pred):
        """计算梯度"""
        return X.T @ (y_pred - y) / len(y)
    
    def fit(self, X, y):
        # 添加偏置项
        X = np.column_stack([np.ones(len(X)), X])
        
        # 初始化权重
        self.weights = np.random.randn(X.shape[1])
        
        losses = []
        for epoch in range(self.epochs):
            # 前向传播
            z = X @ self.weights
            y_pred = self.sigmoid(z)
            
            # 计算损失
            loss = self.cross_entropy_loss(y, y_pred)
            losses.append(loss)
            
            # 反向传播
            gradient = self.gradient(X, y, y_pred)
            self.weights -= self.lr * gradient
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        z = X @ self.weights
        return self.sigmoid(z)
```

### 可视化Loss函数

```python
import matplotlib.pyplot as plt

# 当y=1时
h = np.linspace(0.01, 0.99, 100)
loss_when_y1 = -np.log(h)

# 当y=0时
loss_when_y0 = -np.log(1 - h)

plt.figure(figsize=(10, 6))
plt.plot(h, loss_when_y1, label='y=1: -log(h)')
plt.plot(h, loss_when_y0, label='y=0: -log(1-h)')
plt.xlabel('Predicted Probability h(x)')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 关键洞察

1. **Cross-Entropy = 负对数似然**
2. **最大化似然 ≡ 最小化Cross-Entropy**
3. **梯度简单**: ∇L = Xᵀ(ŷ - y)
4. **学习效率高**: 预测错误时梯度大
5. **统计意义**: 等价于最小化KL散度

### 面试要点

1. LogisticRegression使用Cross-Entropy Loss
2. 从Bernoulli分布 + 最大似然推导
3. 梯度: ∇L = (1/n)Xᵀ(ŷ - y)
4. 不适合用MSE（梯度消失问题）
5. Cross-Entropy = 最小化KL散度
