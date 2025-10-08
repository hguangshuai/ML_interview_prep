#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片生成工具 - 为markdown文件生成示例图片
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_bias_variance_diagram():
    """创建偏差-方差权衡图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 模型复杂度
    complexity = np.linspace(0, 10, 100)
    
    # 偏差（随复杂度增加而减少）
    bias_squared = 2 * np.exp(-complexity/3) + 0.1
    
    # 方差（随复杂度增加而增加）
    variance = 0.1 * complexity + 0.05
    
    # 总误差
    total_error = bias_squared + variance + 0.05
    
    ax.plot(complexity, bias_squared, 'r-', linewidth=2, label='偏差²')
    ax.plot(complexity, variance, 'b-', linewidth=2, label='方差')
    ax.plot(complexity, total_error, 'g-', linewidth=3, label='总误差')
    
    # 找到最优点
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]
    
    ax.axvline(optimal_complexity, color='orange', linestyle='--', alpha=0.7)
    ax.plot(optimal_complexity, optimal_error, 'ro', markersize=10)
    
    ax.set_xlabel('模型复杂度')
    ax.set_ylabel('误差')
    ax.set_title('偏差-方差权衡')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加区域标注
    ax.text(2, 1.5, '欠拟合区域\n(高偏差，低方差)', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(7, 1.5, '过拟合区域\n(低偏差，高方差)', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_overfitting_diagram():
    """创建过拟合/欠拟合图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练数据
    np.random.seed(42)
    x_train = np.linspace(0, 10, 20)
    y_train = 0.5 * x_train + 2 + np.random.normal(0, 1, 20)
    
    # 测试数据
    x_test = np.linspace(0, 10, 100)
    y_test = 0.5 * x_test + 2
    
    # 欠拟合模型（线性）
    ax1.scatter(x_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax1.plot(x_test, y_test, 'g-', linewidth=2, label='真实函数')
    
    # 简单的线性拟合
    poly_coeffs = np.polyfit(x_train, y_train, 1)
    y_pred_simple = np.polyval(poly_coeffs, x_test)
    ax1.plot(x_test, y_pred_simple, 'r--', linewidth=2, label='模型预测')
    
    ax1.set_title('欠拟合 (Underfitting)')
    ax1.set_xlabel('特征')
    ax1.set_ylabel('目标值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 过拟合模型（高次多项式）
    ax2.scatter(x_train, y_train, color='blue', alpha=0.7, label='训练数据')
    ax2.plot(x_test, y_test, 'g-', linewidth=2, label='真实函数')
    
    # 高次多项式拟合
    poly_coeffs_complex = np.polyfit(x_train, y_train, 15)
    y_pred_complex = np.polyval(poly_coeffs_complex, x_test)
    ax2.plot(x_test, y_pred_complex, 'r--', linewidth=2, label='模型预测')
    
    ax2.set_title('过拟合 (Overfitting)')
    ax2.set_xlabel('特征')
    ax2.set_ylabel('目标值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_bayes_theorem_diagram():
    """创建贝叶斯定理可视化图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建韦恩图风格的图
    circle1 = plt.Circle((0.3, 0.5), 0.25, color='lightblue', alpha=0.7)
    circle2 = plt.Circle((0.7, 0.5), 0.25, color='lightcoral', alpha=0.7)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # 添加标签
    ax.text(0.3, 0.5, 'P(Y)', fontsize=16, ha='center', va='center', weight='bold')
    ax.text(0.7, 0.5, 'P(X)', fontsize=16, ha='center', va='center', weight='bold')
    ax.text(0.5, 0.5, 'P(X,Y)', fontsize=14, ha='center', va='center', weight='bold')
    
    # 添加公式
    ax.text(0.5, 0.2, 'P(Y|X) = P(X|Y) × P(Y) / P(X)', 
            fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('贝叶斯定理可视化', fontsize=16, weight='bold')
    
    return fig

def create_generative_vs_discriminative_diagram():
    """创建生成式vs判别式模型对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 生成式模型
    ax1.set_title('生成式模型 (Generative)', fontsize=14, weight='bold')
    ax1.text(0.5, 0.8, '学习 P(X,Y)', fontsize=12, ha='center', va='center')
    ax1.text(0.5, 0.6, '贝叶斯定理', fontsize=12, ha='center', va='center')
    ax1.text(0.5, 0.4, 'P(Y|X) = P(X|Y) × P(Y) / P(X)', fontsize=10, ha='center', va='center')
    ax1.text(0.5, 0.2, '可以生成新样本', fontsize=12, ha='center', va='center')
    
    # 添加箭头
    ax1.arrow(0.2, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    ax1.arrow(0.8, 0.7, -0.1, 0, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 判别式模型
    ax2.set_title('判别式模型 (Discriminative)', fontsize=14, weight='bold')
    ax2.text(0.5, 0.8, '直接学习 P(Y|X)', fontsize=12, ha='center', va='center')
    ax2.text(0.5, 0.6, '学习决策边界', fontsize=12, ha='center', va='center')
    ax2.text(0.5, 0.4, '专注于分类任务', fontsize=12, ha='center', va='center')
    ax2.text(0.5, 0.2, '不能生成新样本', fontsize=12, ha='center', va='center')
    
    # 添加箭头
    ax2.arrow(0.2, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, fc='red', ec='red')
    ax2.arrow(0.8, 0.7, -0.1, 0, head_width=0.02, head_length=0.02, fc='red', ec='red')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """生成所有图片"""
    # 确保图片目录存在
    os.makedirs('images/basic_ml', exist_ok=True)
    os.makedirs('images/generative_models', exist_ok=True)
    
    print("🎨 开始生成图片...")
    
    # 生成基础ML图片
    print("📊 生成偏差-方差权衡图...")
    fig1 = create_bias_variance_diagram()
    fig1.savefig('images/basic_ml/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("📊 生成过拟合/欠拟合图...")
    fig2 = create_overfitting_diagram()
    fig2.savefig('images/basic_ml/overfitting_underfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 生成生成模型图片
    print("🤖 生成贝叶斯定理图...")
    fig3 = create_bayes_theorem_diagram()
    fig3.savefig('images/generative_models/bayes_theorem_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("🤖 生成生成式vs判别式对比图...")
    fig4 = create_generative_vs_discriminative_diagram()
    fig4.savefig('images/generative_models/generative_vs_discriminative_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("✅ 所有图片生成完成！")
    print("📁 图片保存在 images/ 文件夹中")

if __name__ == "__main__":
    main()
