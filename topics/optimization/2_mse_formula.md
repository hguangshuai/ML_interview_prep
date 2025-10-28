# MSE公式解释及使用场景

## MSE的定义

**均方误差(Mean Squared Error, MSE)** 是回归问题中最常用的损失函数之一。

### 数学定义

```python
MSE = (1/n) * Σ(y_i - ŷ_i)²
```

其中：
- n: 样本数量
- y_i: 第i个真实值
- ŷ_i: 第i个预测值

### MSE的性质

1. **非负性**: MSE ≥ 0
2. **对异常值敏感**: 平方惩罚对大误差更严厉
3. **可微性**: 处处可微，适合梯度下降
4. **凸性**: 凸函数，保证全局最优

### 何时使用MSE？

#### ✅ 适合的场景

- **回归问题**（连续值预测）
- **误差服从正态分布**
- **需要惩罚大误差**的应用
- **需要可微loss**用于反向传播

#### ❌ 不适合的场景

- **存在异常值**（考虑MAE或Huber Loss）
- **误差分布不是正态分布**
- **需要直观可解释**（MAE更直观）

### MSE vs 其他Loss

**MSE vs MAE:**
```python
MSE = (1/n) * Σ(y - ŷ)²      # 平方惩罚
MAE = (1/n) * Σ|y - ŷ|       # 绝对惩罚
```

**特点对比:**

| 特性 | MSE | MAE |
|------|-----|-----|
| 对异常值敏感 | ✅ 高 | ❌ 低 |
| 可微性 | ✅ 处处可微 | ⚠️ 在0处不可微 |
| 梯度 | 2(y - ŷ) | ±1 |
| 凸性 | ✅ | ✅ |

### 代码示例

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def mse(y_true, y_pred):
    """计算MSE"""
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(mse(y_true, y_pred))

# 示例
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.2, 2.1, 2.9, 4.2, 4.8])

print(f"MSE: {mse(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
```

### RMSE

**Root Mean Squared Error:**
```python
RMSE = √MSE
```

- MSE的单位是平方单位
- RMSE是原始单位，更易解释

### 在深度学习中的应用

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # 回归输出
])

model.compile(
    optimizer='adam',
    loss='mse',  # MSE损失函数
    metrics=['mae', 'mse']
)
```

### 面试要点

1. MSE = (1/n) * Σ(y - ŷ)²
2. 对异常值敏感，但可微且凸
3. 适用于回归问题和正态误差分布
4. RMSE = √MSE，更易解释
