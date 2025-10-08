# AI/MLE 面试准备项目

这是一个全面的AI/机器学习工程师面试准备项目，采用简洁的主题分类结构，方便复习和学习。

## 📁 项目结构

```
ML design/
├── topics/                    # 🎯 按技术主题分类（所有相关内容集中）
│   ├── basic_ml/             # 基础机器学习
│   │   ├── *.md              # 问题+答案文档
│   │   └── *.png             # 相关图片
│   ├── generative_models/    # 生成模型
│   │   ├── *.md              # 问题+答案文档
│   │   ├── coding_challenges/ # 编程挑战
│   │   └── images/           # 相关图片
│   ├── deep_learning/        # 深度学习
│   ├── nlp/                  # 自然语言处理
│   ├── computer_vision/      # 计算机视觉
│   ├── ml_system_design/     # 机器学习系统设计
│   └── statistics/           # 统计学
├── projects/                  # 🚀 完整项目
│   └── pipeline_projects/    # ML Pipeline项目
├── tools/                     # 🛠️ 练习工具
│   ├── question_generator.py # 随机问题生成器
│   └── quiz_system.py        # 测验系统
├── examples/                  # 📊 可视化示例
├── interview_guides/          # 📖 面试指南
├── resources/                 # 📚 学习资源
└── images/                   # 🖼️ 共享图片资源
```

## 📚 学习资料导航

### 🎯 按技术主题分类

#### 📊 基础机器学习 (Basic ML)
- [过拟合/欠拟合详解](topics/basic_ml/overfitting_underfitting.md) - 核心概念、检测方法、解决方案
- [偏差-方差权衡](topics/basic_ml/bias_variance_tradeoff.md) - 数学原理、四种状态、实际应用
- [过拟合预防手段](topics/basic_ml/overfitting_prevention.md) - 正则化、早停法、Dropout等

#### 🤖 生成模型 (Generative Models)
- [生成式 vs 判别式模型](topics/generative_models/generative_vs_discriminative.md) - 贝叶斯定理、实际对比

#### 🧠 深度学习 (Deep Learning)
- [深度学习基础概念](topics/deep_learning/basic_concepts.md) - 梯度消失、注意力机制、Transformer优势

#### 🔤 自然语言处理 (NLP)
- 即将添加...

#### 👁️ 计算机视觉 (Computer Vision)
- 即将添加...

#### 🏗️ 机器学习系统设计 (ML System Design)
- 即将添加...

#### 📈 统计学 (Statistics)
- 即将添加...

### 🛠️ 实用工具

#### 📊 可视化示例
- [偏差-方差可视化](examples/bias_variance_visualization.py) - 生成分析图表
- [过拟合/欠拟合可视化](examples/overfitting_underfitting_visualization.py) - 学习曲线分析

#### 🎲 练习工具
- [随机问题生成器](tools/question_generator.py) - 智能出题系统
- [测验系统](tools/quiz_system.py) - 交互式练习

#### 📖 面试指南
- [过拟合/欠拟合面试指南](interview_guides/overfitting_underfitting_guide.md) - 面试准备要点

### 🚀 完整项目

#### 🔄 ML Pipeline项目
- [完整ML Pipeline](projects/pipeline_projects/) - 端到端机器学习项目

#### 💻 编程挑战
- 编程挑战已移除，专注于理论学习

## ✨ 新结构优势

### 🎯 **主题集中**
- 每个技术主题的所有内容（问题、答案、代码、图片）都在同一个文件夹
- 复习时只需要关注一个文件夹，不会遗漏任何相关内容

### 📚 **学习友好**
- 按技术领域分类，便于系统性学习
- 每个主题独立，可以按需深入某个领域

### 🔍 **查找便捷**
- 不再有重复的文件夹结构
- 清晰的层次结构，一目了然

## 🚀 使用方法

### 1. 学习特定主题
```bash
# 学习生成模型相关内容
cd topics/generative_models/
ls -la  # 查看所有相关文件

# 阅读问题和答案
cat generative_vs_discriminative.md
```

### 2. 随机练习
```bash
# 运行问题生成器
python tools/question_generator.py

# 开始测验
python tools/quiz_system.py
```

### 3. 可视化学习
```bash
# 运行可视化示例
python examples/overfitting_underfitting_visualization.py
```

## 📖 学习路径建议

### 阶段1: 基础巩固 (1-2周)
```
topics/basic_ml/          # 基础机器学习概念
topics/statistics/        # 统计学基础
```

### 阶段2: 深度学习 (2-3周)
```
topics/deep_learning/     # 深度学习理论
topics/generative_models/ # 生成模型理论
```

### 阶段3: 专业领域 (按需)
```
topics/nlp/               # 自然语言处理
topics/computer_vision/   # 计算机视觉
topics/ml_system_design/  # 系统设计
```

### 阶段4: 实践项目 (持续)
```
projects/                 # 完整项目实践
tools/                    # 练习工具使用
```

## 🎯 主题内容说明

### 📚 每个主题文件夹包含：
- **问题文档** (`*.md`): 面试问题和详细答案
- **编程挑战** (`coding_challenges/`): 相关代码实现
- **图片资源** (`images/` 或 `*.png`): 可视化图表
- **学习资源**: 相关论文、教程链接等

### 🔧 编程挑战包含：
- 完整实现代码
- 测试套件
- 详细说明文档
- 性能基准

## 📝 开发规范

### 文件命名规范
- 问题文档: `topic_name.md`
- 图片文件: `descriptive_name.png`
- 代码文件: 使用描述性名称

### 内容结构规范
每个问题文档应包含：
- 中文理解（便于记忆）
- 英文标准答案（面试回答）
- 数学公式和代码示例
- 可视化图表
- 面试follow-up问题

## 🔄 更新和维护

### 添加新主题
1. 在 `topics/` 下创建新文件夹
2. 添加问题文档和编程挑战
3. 更新本README文件

### 添加新问题
1. 在对应主题文件夹下创建 `.md` 文件
2. 按规范格式编写内容
3. 添加相关图片和代码

## 🎉 开始学习

选择你感兴趣的主题，进入对应文件夹开始学习：

```bash
# 查看所有可用主题
ls topics/

# 进入感兴趣的主题
cd topics/[your_topic]/

# 开始学习
ls -la  # 查看所有相关文件
```

---

**🎯 现在开始你的AI/MLE面试准备之旅！每个主题都是完整的学习单元，让你高效复习！** 🚀