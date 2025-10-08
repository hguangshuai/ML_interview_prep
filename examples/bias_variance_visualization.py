"""
偏差-方差权衡可视化分析
生成详细的偏差-方差权衡图表和示例
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n_samples = 200
    
    # 真实函数: y = 0.5x + sin(x) + noise
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 0.5 * X.flatten() + np.sin(X.flatten()) + np.random.normal(0, 0.3, n_samples)
    
    return X, y

def create_models():
    """创建不同复杂度的模型"""
    models = {}
    
    # 1. 线性模型（高偏差，低方差）
    models['Linear'] = LinearRegression()
    
    # 2. 多项式模型（中偏差，中方差）
    models['Polynomial'] = LinearRegression()
    
    # 3. 随机森林（低偏差，中方差）
    models['Random Forest'] = RandomForestRegressor(n_estimators=50, random_state=42)
    
    return models

def plot_bias_variance_detailed():
    """绘制详细的偏差-方差分析"""
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = create_models()
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('偏差-方差权衡详细分析', fontsize=16, fontweight='bold')
    
    # 1. 线性模型
    ax1 = axes[0, 0]
    model = models['Linear']
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    ax1.scatter(X_train, y_train, alpha=0.6, label='训练数据', color='blue')
    ax1.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='red')
    
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    ax1.plot(X_plot, y_plot, 'g-', linewidth=2, label='模型预测')
    
    ax1.set_title(f'线性模型 (高偏差，低方差)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 多项式模型
    ax2 = axes[0, 1]
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = models['Polynomial']
    model.fit(X_train_poly, y_train)
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    ax2.scatter(X_train, y_train, alpha=0.6, label='训练数据', color='blue')
    ax2.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='red')
    
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    ax2.plot(X_plot, y_plot, 'g-', linewidth=2, label='模型预测')
    
    ax2.set_title(f'多项式模型 (中偏差，中方差)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 随机森林
    ax3 = axes[0, 2]
    model = models['Random Forest']
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    ax3.scatter(X_train, y_train, alpha=0.6, label='训练数据', color='blue')
    ax3.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='red')
    
    y_plot = model.predict(X_plot)
    ax3.plot(X_plot, y_plot, 'g-', linewidth=2, label='模型预测')
    
    ax3.set_title(f'随机森林 (低偏差，中方差)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 偏差-方差权衡图
    ax4 = axes[1, 0]
    complexity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bias_squared = np.exp(-complexity/3) + 0.1
    variance = complexity * 0.1 + 0.05
    noise = 0.1
    total_error = bias_squared + variance + noise
    
    ax4.plot(complexity, bias_squared, 'b-', linewidth=2, label='偏差² (Bias²)')
    ax4.plot(complexity, variance, 'r-', linewidth=2, label='方差 (Variance)')
    ax4.plot(complexity, total_error, 'g-', linewidth=3, label='总误差 (Total Error)')
    ax4.axhline(y=noise, color='gray', linestyle='--', alpha=0.7, label='噪声 (Noise)')
    
    optimal_idx = np.argmin(total_error)
    ax4.axvline(x=complexity[optimal_idx], color='orange', linestyle='--', alpha=0.7)
    ax4.plot(complexity[optimal_idx], total_error[optimal_idx], 'o', color='orange', markersize=8)
    
    ax4.set_xlabel('模型复杂度 (Model Complexity)')
    ax4.set_ylabel('误差 (Error)')
    ax4.set_title('偏差-方差权衡 (Bias-Variance Tradeoff)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 学习曲线
    ax5 = axes[1, 1]
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        n_samples = int(size * len(X_train))
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        model = LinearRegression()
        model.fit(X_subset, y_subset)
        
        train_score = model.score(X_subset, y_subset)
        val_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    ax5.plot(train_sizes, train_scores, 'o-', label='训练分数', color='blue')
    ax5.plot(train_sizes, val_scores, 'o-', label='验证分数', color='red')
    ax5.set_title('学习曲线 (Learning Curve)')
    ax5.set_xlabel('训练集大小比例')
    ax5.set_ylabel('R² 分数')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 模型复杂度对比
    ax6 = axes[1, 2]
    model_names = ['Linear', 'Polynomial', 'Random Forest']
    bias_values = [0.8, 0.3, 0.1]
    variance_values = [0.1, 0.3, 0.4]
    total_errors = [b + v + 0.1 for b, v in zip(bias_values, variance_values)]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax6.bar(x - width, bias_values, width, label='偏差²', alpha=0.8)
    ax6.bar(x, variance_values, width, label='方差', alpha=0.8)
    ax6.bar(x + width, total_errors, width, label='总误差', alpha=0.8)
    
    ax6.set_xlabel('模型类型')
    ax6.set_ylabel('误差值')
    ax6.set_title('不同模型的偏差-方差对比')
    ax6.set_xticks(x)
    ax6.set_xticklabels(model_names)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/basic_ml/bias_variance_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_complexity_analysis():
    """绘制模型复杂度分析"""
    X, y = generate_sample_data()
    
    # 不同复杂度的多项式模型
    degrees = [1, 2, 3, 5, 10, 15]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型复杂度对偏差-方差的影响', fontsize=16, fontweight='bold')
    
    for i, degree in enumerate(degrees):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 多项式特征
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # 计算误差
        mse = mean_squared_error(y, y_pred)
        
        # 绘制结果
        ax.scatter(X, y, alpha=0.6, label='真实数据', color='blue')
        ax.plot(X, y_pred, 'r-', linewidth=2, label=f'模型预测 (degree={degree})')
        
        ax.set_title(f'多项式回归 (degree={degree})\nMSE: {mse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/basic_ml/model_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_bias_variance_summary():
    """创建偏差-方差总结表"""
    import pandas as pd
    
    summary_data = {
        '模型类型': ['线性回归', '多项式回归', '随机森林', '深度网络'],
        '偏差': ['高', '中', '低', '低'],
        '方差': ['低', '中', '中', '高'],
        '适用场景': [
            '线性关系明显',
            '非线性但不太复杂',
            '复杂非线性关系',
            '非常复杂的关系'
        ],
        '优缺点': [
            '简单稳定，但可能欠拟合',
            '平衡性好，需要调参',
            '性能好，可解释性中等',
            '性能最好，但容易过拟合'
        ]
    }
    
    df = pd.DataFrame(summary_data)
    print("\n📊 偏差-方差权衡总结表:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return df

def plot_regularization_effect():
    """绘制正则化对偏差-方差的影响"""
    from sklearn.linear_model import Ridge, Lasso
    
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 高次多项式特征
    poly_features = PolynomialFeatures(degree=10)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # 不同正则化强度
    alphas = [0, 0.01, 0.1, 1, 10, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('正则化对偏差-方差的影响', fontsize=16, fontweight='bold')
    
    for i, alpha in enumerate(alphas):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Ridge回归
        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        ax.scatter(X_train, y_train, alpha=0.6, label='训练数据', color='blue')
        ax.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='red')
        
        X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
        X_plot_poly = poly_features.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        ax.plot(X_plot, y_plot, 'g-', linewidth=2, label='模型预测')
        
        ax.set_title(f'Ridge回归 (α={alpha})\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/basic_ml/regularization_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    print("🎯 偏差-方差权衡可视化分析")
    print("=" * 50)
    
    # 生成详细分析图
    plot_bias_variance_detailed()
    
    # 生成模型复杂度分析图
    plot_model_complexity_analysis()
    
    # 生成正则化效果图
    plot_regularization_effect()
    
    # 创建总结表
    summary_df = create_bias_variance_summary()
    
    print("\n✅ 分析完成！图表已保存为 PNG 文件")
    print("📁 文件: images/basic_ml/bias_variance_detailed_analysis.png")
    print("📁 文件: images/basic_ml/model_complexity_analysis.png")
    print("📁 文件: images/basic_ml/regularization_effect.png")
