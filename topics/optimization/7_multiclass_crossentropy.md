# 多分类Cross-Entropy

## English Interview Answer
For multi‑class logistic regression, the Softmax + Cross‑Entropy loss is used: L = −Σ y_k log ŷ_k. It is the negative log‑likelihood under a categorical (multinomial) model, equivalent to minimizing KL(y||ŷ), with stable gradients and probabilistic interpretation.

## 中文知识点解释（含英文术语标注）
- Softmax：将 logits 变为概率分布（probability distribution）。
- 多分类交叉熵（Multi‑Class Cross‑Entropy）：L = −Σ y_k log ŷ_k，对应类别分布的负对数似然（negative log‑likelihood）。
- 等价性：最小化交叉熵 ⇔ 最小化 KL(y||ŷ) ⇔ 最大似然（Maximum Likelihood）。
- 优点：梯度稳定（stable gradients）、有概率解释（probabilistic interpretation）。

## 核心答案

**Multi-Class LogisticRegression使用Multi-Class Cross-Entropy Loss**

## 详细推导

### 1. Softmax函数

**多分类扩展Sigmoid:**
```python
P(Y=k|X) = exp(W_k·X) / Σ exp(W_j·X)
```

其中：
- k: 第k个类别
- W_k: 第k个类别的权重向量

**向量形式:**
```python
P = softmax(logits) = softmax(W @ X)
```

### 2. Multi-Class Cross-Entropy

**标签编码:**
```python
# One-hot encoding
y = [0, 0, 1, 0]  # 第3类是正类
```

**Loss函数:**
```python
L = -Σ y_i * log(ŷ_i)
  = -log(ŷ_k)  # k是正确的类别
```

其中ŷ_k是k类的预测概率。

### 3. 为什么用Cross-Entropy？

#### 从最大似然估计推导

**假设Y服从Multinomial分布:**
```python
P(Y=k|X) = p_k
```

**似然函数:**
```python
L(W) = Π P(y_i | x_i, W) 
     = Π softmax(W @ x_i)[y_i]
```

**对数似然:**
```python
l(W) = Σ log(P(y_i | x_i, W))
     = Σ log(softmax(W @ x_i)[y_i])
     = Σ [W_yi @ x_i - log(Σ exp(W_j @ x_i))]
```

**最大化似然 ≡ 最小化负对数似然:**
```python
L(W) = -Σ log(softmax(W @ x_i)[y_i])
     = Σ Cross-Entropy(y_i, ŷ_i)
```

#### 从KL散度推导

**训练目标: 最小化KL散度**
```python
min KL(P||Q) = min Σ p_k log(p_k/q_k)
            = min -Σ p_k log(q_k)  # p_k log(p_k)是常数
            = min -Σ log(q_k)  # 其中k是正确的类别
            = min Cross-Entropy
```

### 4. 完整推导

**One-hot标签:**
```python
y = [y_1, y_2, ..., y_K]
其中: Σ y_k = 1, y_k ∈ {0, 1}
```

**预测概率:**
```python
ŷ = softmax(logits) = [ŷ_1, ŷ_2, ..., ŷ_K]
```

**Cross-Entropy Loss:**
```python
L = -Σ y_k * log(ŷ_k)
  = -log(ŷ_true_class)
```

### 5. 代码实现

```python
import numpy as np

def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(y_true, y_pred):
    """Multi-Class Cross-Entropy Loss"""
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))

# 示例：3分类问题
num_classes = 3
num_samples = 5

# 真实标签（one-hot编码）
y_true = np.array([
    [1, 0, 0],  # 类别0
    [0, 1, 0],  # 类别1
    [0, 0, 1],  # 类别2
    [1, 0, 0],
    [0, 1, 0]
])

# 模型预测（logits）
logits = np.random.randn(num_samples, num_classes)

# 转换为概率
y_pred = softmax(logits)

# 计算损失
loss = cross_entropy(y_true, y_pred)
print(f"Cross-Entropy Loss: {loss:.4f}")
```

### 6. 梯度推导

**对第k个类别权重W_k求导:**
```python
∂L/∂W_k = (1/n) * Σ x * (ŷ_k - y_k)
```

**梯度解释:**
- 如果预测概率ŷ_k > 真实标签y_k，则减少该类的权重
- 如果预测概率ŷ_k < 真实标签y_k，则增加该类的权重

**梯度矩阵:**
```python
∇L = (1/n) * Xᵀ(ŷ - y)
```

### 7. 为什么要用Cross-Entropy？

#### 1. 概率解释

Cross-Entropy最小化预测分布与真实分布之间的KL散度：
```python
min KL(y_true || y_pred) = min Cross-Entropy
```

#### 2. 梯度特性

**Cross-Entropy的梯度:**
```python
∂L/∂W_k = (1/n) * Xᵀ(ŷ_k - y_k)
```

- 当预测错误时，梯度大
- 当预测正确时，梯度小
- 学习效率高

#### 3. 信息论基础

Cross-Entropy是**最小化分布间的差异**的自然选择。

#### 4. 与其他Loss对比

| Loss Function | 适用场景 | 特点 |
|--------------|---------|------|
| **Cross-Entropy** | 多分类 | 概率解释，梯度友好 |
| MSE | 不适合 | 非凸，梯度消失 |
| Hinge Loss | 二元分类 | 稀疏性 |

### 8. 数值稳定性

```python
def stable_softmax(x):
    """数值稳定的Softmax"""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def stable_cross_entropy(y_true, y_pred):
    """数值稳定的Cross-Entropy"""
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))
```

### 9. 实际应用

```python
import tensorflow as tf

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10分类
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Multi-Class Cross-Entropy
    metrics=['accuracy']
)

# 训练
model.fit(X_train, y_train_onehot, epochs=10)
```

### 10. 关键洞察

**Cross-Entropy的优势:**
1. **概率解释**: 最小化预测分布与真实分布的距离
2. **梯度有效**: 错误预测时梯度大
3. **理论支撑**: 等价于最大似然估计和KL散度最小化
4. **适用性**: 适用于多分类、标签平滑等场景

### 面试要点

1. **Multi-Class Cross-Entropy**: L = -Σ y_k * log(ŷ_k)
2. **Softmax激活函数**: 将logits转换为概率
3. **One-hot编码**: y_true必须是one-hot
4. **等价性**: 最小化Cross-Entropy = 最小化KL散度 = 最大化似然
5. **为什么用它**: 概率解释、梯度友好、理论完备
