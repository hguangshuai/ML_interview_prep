# ML Coding Practice Notebooks

这个文件夹包含了按主题分类的机器学习编程练习notebooks，帮助你准备ML相关的coding面试。

## 📁 文件结构

```
ml_coding_practice/
├── core_algorithms.ipynb           # 核心算法实现
├── metrics_evaluation.ipynb        # 评估指标和交叉验证
├── vectorization.ipynb            # 向量化和数据处理
├── recommendation.ipynb           # 推荐系统和检索
├── deep_learning.ipynb            # 深度学习和Transformer
├── system_design.ipynb            # 系统设计和工程实践
└── README.md                      # 本文件
```

## 📚 各主题包含内容

### 1. Core Algorithms (`core_algorithms.ipynb`)
- Linear Regression (梯度下降 + L2正则化)
- Logistic Regression (交叉熵 + Sigmoid)
- K-Nearest Neighbors (向量化距离计算)
- Naive Bayes (高斯朴素贝叶斯)
- Decision Tree (ID3/CART + Gini/Entropy)
- K-Means Clustering (K-means++初始化)
- PCA (SVD分解 + 解释方差)
- Neural Network (2层 + 反向传播)

### 2. Metrics & Evaluation (`metrics_evaluation.ipynb`)
- 混淆矩阵构建
- Precision, Recall, F1计算
- ROC曲线和AUC计算
- PR曲线和平均精度
- K折交叉验证
- 分层采样

### 3. Vectorization (`vectorization.ipynb`)
- NumPy向量化操作
- 数据标准化和距离计算
- 余弦相似度矩阵
- Pandas数据处理
- 分组聚合和缺失值处理

### 4. Recommendation (`recommendation.ipynb`)
- 矩阵分解 (SGD)
- TF-IDF + 余弦相似度
- 检索评估指标 (Recall@K, MRR, MAP)

### 5. Deep Learning (`deep_learning.ipynb`)
- CNN从零实现
- Graph Neural Networks基础
- Transformer注意力机制
- 位置编码

### 6. System Design (`system_design.ipynb`)
- ML Pipeline设计
- 在线预测系统
- A/B测试框架

## 🎯 练习建议

### 第1周练习计划
- **Day 1**: 核心算法 - Linear & Logistic Regression
- **Day 2**: 核心算法 - KMeans & PCA  
- **Day 3**: 核心算法 - Neural Network (2-layer)
- **Day 4**: 评估指标 - Metrics & Cross Validation
- **Day 5**: 推荐系统 - Cosine Similarity & Recall@K
- **Day 6**: 深度学习 - Attention & Chunking
- **Day 7**: 系统设计 - End-to-end Pipeline

### 练习要点
1. **纯Python/NumPy实现** - 不使用sklearn/pytorch等框架
2. **完整的方法** - 实现fit、predict、score等方法
3. **数值稳定性** - 注意梯度检查、softmax稳定性等
4. **向量化计算** - 避免循环，使用NumPy广播
5. **测试验证** - 与标准实现对比验证正确性

## 📚 参考文档
- [ML编程面试准备指南](../ML_coding.md) - 完整的编程面试准备指南
- [项目README](../README.md) - 项目整体结构和使用方法

## 🚀 开始练习

选择你感兴趣的主题，打开对应的notebook开始编程练习：

```bash
# 查看所有可用notebooks
ls ml_coding_practice/*.ipynb

# 进入文件夹
cd ml_coding_practice/

# 打开感兴趣的notebook开始练习
jupyter notebook core_algorithms.ipynb
jupyter notebook metrics_evaluation.ipynb
jupyter notebook vectorization.ipynb
jupyter notebook recommendation.ipynb
jupyter notebook deep_learning.ipynb
jupyter notebook system_design.ipynb
```

---

**💡 提示**: 每个notebook都包含了详细的实现要求、代码框架和测试用例，帮助你系统地练习ML编程技能！
