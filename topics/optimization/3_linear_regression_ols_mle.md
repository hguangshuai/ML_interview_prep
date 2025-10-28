# LinearRegression最小二乘法和MLE的关系

## 核心答案

**在正态分布误差的假设下，最小二乘法(OLS)等价于最大似然估计(MLE)。**

## 详细推导

### OLS推导

**目标:** 最小化残差平方和

```python
L(β) = Σ(y_i - X_iβ)² = (y - Xβ)ᵀ(y - Xβ)
```

**求解:**
```python
∂L/∂β = -2Xᵀ(y - Xβ) = 0
XᵀXβ = Xᵀy
β̂ = (XᵀX)⁻¹Xᵀy
```

### MLE推导

**假设:** ε ~ N(0, σ²I)

**似然函数:**
```python
L(β, σ²) = Π p(y_i | X_i, β, σ²)
```

**对数似然:**
```python
ln L(β, σ²) = -n/2 * ln(2πσ²) - (1/(2σ²)) * Σ(y_i - X_iβ)²
```

**最大化似然:**
```python
max ln L(β, σ²) ⇔ min Σ(y_i - X_iβ)²
```

由于σ² > 0，最大化ln L等价于最小化SSE！

**求解:**
```python
β̂_MLE = (XᵀX)⁻¹Xᵀy = β̂_OLS
```

### 等价性证明

| 方法 | 目标函数 | 假设 | 结果 |
|------|---------|------|------|
| **OLS** | min Σ(y - Xβ)² | 线性关系 | β̂ |
| **MLE** | max ln L(β,σ²) | 正态误差 | β̂ |

两者在正态误差假设下得到**相同的β估计值**！

### σ²的估计

**MLE估计（有偏）:**
```python
σ²̂_MLE = (1/n) * Σ(y - Xβ̂)²
```

**OLS估计（无偏）:**
```python
σ²̂_OLS = (1/(n-p)) * Σ(y - Xβ̂)²
```

### 代码验证

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 生成数据
np.random.seed(42)
n = 100
X = np.random.randn(n, 3)
true_beta = np.array([2, -1, 0.5])
y = X @ true_beta + np.random.normal(0, 1, n)

# OLS
model = LinearRegression()
model.fit(X, y)
beta_ols = model.coef_

print(f"OLS β: {beta_ols}")

# MLE目标函数
from scipy.optimize import minimize

def negative_log_likelihood(params, X, y):
    beta = params[:-1]
    sigma2 = params[-1]
    residuals = y - X @ beta
    nll = n/2 * np.log(2 * np.pi * sigma2) + np.sum(residuals**2) / (2 * sigma2)
    return nll

initial_params = np.concatenate([np.zeros(3), [1.0]])
result = minimize(negative_log_likelihood, initial_params, args=(X, y))
beta_mle = result.x[:-1]

print(f"MLE β: {beta_mle}")
print(f"Difference: {np.abs(beta_ols - beta_mle)}")  # 接近0
```

### 为什么重要？

1. **统计理论基础**: MLE提供一致性、渐近正态性
2. **置信区间**: 可以构建置信区间和假设检验
3. **模型诊断**: 检验正态分布假设

### 面试要点

1. OLS和MLE在正态误差假设下等价
2. 两者得到相同的β估计
3. σ²估计不同（自由度校正）
4. MLE提供统计推断的理论基础
