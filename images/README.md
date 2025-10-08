# 📸 图片资源管理

## 📁 文件夹结构

```
images/
├── basic_ml/                    # 基础机器学习相关图片
│   ├── bias_variance_tradeoff.png
│   ├── bias_variance_detailed_analysis.png
│   ├── model_complexity_analysis.png
│   ├── overfitting_underfitting_analysis.png
│   └── regularization_effect.png
├── generative_models/           # 生成模型相关图片
│   ├── generative_vs_discriminative_comparison.png
│   ├── bayes_theorem_visualization.png
│   ├── decision_boundary_comparison.png
│   └── data_efficiency_comparison.png
├── deep_learning/               # 深度学习相关图片
├── nlp/                         # 自然语言处理相关图片
├── computer_vision/             # 计算机视觉相关图片
├── ml_system_design/            # 机器学习系统设计相关图片
└── statistics/                  # 统计学相关图片
```

## 🎨 如何添加图片

### 1. 图片命名规范
- 使用描述性名称，如：`bias_variance_tradeoff.png`
- 使用下划线分隔单词
- 包含主题关键词

### 2. 图片格式
- 推荐使用PNG格式（支持透明背景）
- 也可以使用JPG格式
- 确保图片清晰度足够

### 3. 在Markdown中引用图片
```markdown
![图片描述](../../images/主题文件夹/图片名称.png)
```

### 4. 图片尺寸建议
- 宽度：800-1200像素
- 高度：根据内容调整
- 保持长宽比协调

## 📊 现有图片说明

### Basic ML 图片
- `bias_variance_tradeoff.png` - 偏差-方差权衡基础图
- `bias_variance_detailed_analysis.png` - 详细分析图
- `model_complexity_analysis.png` - 模型复杂度分析
- `overfitting_underfitting_analysis.png` - 过拟合/欠拟合分析
- `regularization_effect.png` - 正则化效果图

### Generative Models 图片（待添加）
- `generative_vs_discriminative_comparison.png` - 两种模型对比
- `bayes_theorem_visualization.png` - 贝叶斯定理可视化
- `decision_boundary_comparison.png` - 决策边界对比
- `data_efficiency_comparison.png` - 数据效率对比

## 🛠️ 生成图片的工具

### Python可视化
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 保存图片
plt.savefig('../../images/basic_ml/your_image_name.png', 
            dpi=300, bbox_inches='tight')
```

### 在线工具
- [Draw.io](https://app.diagrams.net/) - 流程图和架构图
- [Canva](https://www.canva.com/) - 设计图表
- [Excalidraw](https://excalidraw.com/) - 手绘风格图表

## 📝 注意事项

1. **路径正确性**：确保markdown中的图片路径正确
2. **文件大小**：控制图片文件大小，避免过大
3. **版权问题**：使用自己创建或开源图片
4. **一致性**：保持图片风格和颜色一致
5. **可访问性**：为图片添加alt文本描述

## 🔄 更新流程

1. 将图片文件放入对应主题文件夹
2. 在markdown文件中添加图片引用
3. 测试图片显示是否正常
4. 提交到GitHub验证效果