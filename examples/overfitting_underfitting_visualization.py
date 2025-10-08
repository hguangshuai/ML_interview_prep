"""
过拟合和欠拟合可视化示例
帮助理解这两个重要概念
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    # 真实函数: y = 0.5x + sin(x) + noise
    y = 0.5 * X.flatten() + np.sin(X.flatten()) + np.random.normal(0, 0.3, 100)
    return X, y

def create_models():
    """创建不同复杂度的模型"""
    models = {}
    
    # 欠拟合: 线性模型
    models['underfitting'] = LinearRegression()
    
    # 合适拟合: 3次多项式
    models['good_fit'] = LinearRegression()
    
    # 过拟合: 15次多项式
    models['overfitting'] = LinearRegression()
    
    return models

def plot_overfitting_underfitting():
    """绘制过拟合和欠拟合的对比图"""
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = create_models()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('过拟合 vs 欠拟合 对比分析', fontsize=16, fontweight='bold')
    
    # 1. 欠拟合模型
    ax1 = axes[0, 0]
    model = models['underfitting']
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
    
    ax1.set_title(f'欠拟合 (Underfitting)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 合适拟合模型
    ax2 = axes[0, 1]
    poly_features = PolynomialFeatures(degree=3)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = models['good_fit']
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
    
    ax2.set_title(f'合适拟合 (Good Fit)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 过拟合模型
    ax3 = axes[1, 0]
    poly_features = PolynomialFeatures(degree=15)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = models['overfitting']
    model.fit(X_train_poly, y_train)
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    ax3.scatter(X_train, y_train, alpha=0.6, label='训练数据', color='blue')
    ax3.scatter(X_test, y_test, alpha=0.6, label='测试数据', color='red')
    
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    ax3.plot(X_plot, y_plot, 'g-', linewidth=2, label='模型预测')
    
    ax3.set_title(f'过拟合 (Overfitting)\n训练MSE: {train_mse:.3f}, 测试MSE: {test_mse:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 学习曲线对比
    ax4 = axes[1, 1]
    
    # 生成不同训练集大小的学习曲线
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        n_samples = int(size * len(X_train))
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        # 简单线性模型
        model = LinearRegression()
        model.fit(X_subset, y_subset)
        
        train_score = model.score(X_subset, y_subset)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    ax4.plot(train_sizes, train_scores, 'o-', label='训练分数', color='blue')
    ax4.plot(train_sizes, test_scores, 'o-', label='测试分数', color='red')
    ax4.set_title('学习曲线 (Learning Curve)')
    ax4.set_xlabel('训练集大小比例')
    ax4.set_ylabel('R² 分数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/basic_ml/overfitting_underfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'underfitting': {'train_mse': train_mse, 'test_mse': test_mse},
        'good_fit': {'train_mse': train_mse, 'test_mse': test_mse},
        'overfitting': {'train_mse': train_mse, 'test_mse': test_mse}
    }

def plot_bias_variance_tradeoff():
    """绘制偏差-方差权衡图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 模型复杂度
    complexity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # 偏差 (Bias) - 随着复杂度增加而减少
    bias = np.exp(-complexity/3) + 0.1
    
    # 方差 (Variance) - 随着复杂度增加而增加
    variance = complexity * 0.1 + 0.05
    
    # 总误差 = 偏差² + 方差 + 噪声
    noise = 0.1
    total_error = bias**2 + variance + noise
    
    ax.plot(complexity, bias**2, 'b-', linewidth=2, label='偏差² (Bias²)')
    ax.plot(complexity, variance, 'r-', linewidth=2, label='方差 (Variance)')
    ax.plot(complexity, total_error, 'g-', linewidth=3, label='总误差 (Total Error)')
    ax.axhline(y=noise, color='gray', linestyle='--', alpha=0.7, label='噪声 (Noise)')
    
    # 标记最优复杂度
    optimal_idx = np.argmin(total_error)
    ax.axvline(x=complexity[optimal_idx], color='orange', linestyle='--', alpha=0.7)
    ax.plot(complexity[optimal_idx], total_error[optimal_idx], 'o', color='orange', markersize=8)
    
    ax.set_xlabel('模型复杂度 (Model Complexity)')
    ax.set_ylabel('误差 (Error)')
    ax.set_title('偏差-方差权衡 (Bias-Variance Tradeoff)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/basic_ml/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table():
    """创建总结表格"""
    import pandas as pd
    
    summary_data = {
        '特征': ['训练误差', '测试误差', '泛化能力', '模型复杂度', '解决方案'],
        '欠拟合 (Underfitting)': [
            '高', '高', '差', '过低', '增加复杂度、特征工程'
        ],
        '合适拟合 (Good Fit)': [
            '低', '低', '好', '适中', '保持现状'
        ],
        '过拟合 (Overfitting)': [
            '低', '高', '差', '过高', '正则化、增加数据'
        ]
    }
    
    df = pd.DataFrame(summary_data)
    print("\n📊 过拟合/欠拟合对比总结:")
    print("=" * 60)
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    print("🎯 过拟合和欠拟合可视化分析")
    print("=" * 50)
    
    # 生成对比图
    results = plot_overfitting_underfitting()
    
    # 生成偏差-方差权衡图
    plot_bias_variance_tradeoff()
    
    # 创建总结表格
    summary_df = create_summary_table()
    
    print("\n✅ 分析完成！图表已保存为 PNG 文件")
    print("📁 文件: images/basic_ml/overfitting_underfitting_analysis.png")
    print("📁 文件: images/basic_ml/bias_variance_tradeoff.png")
