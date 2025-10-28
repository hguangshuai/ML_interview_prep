# 🛠️ 开发规范文档（Topics-first）

## 📋 项目开发规范

本项目采用“按主题归档”的内容组织方式：所有问题与答案、配图、示例代码都放在 `topics/{topic}/` 子目录中；不再使用 `questions/` 与 `answers/` 的双目录结构。

### 1. 文件组织结构

```
ML design/
└── topics/
    ├── basic_ml/
    │   ├── overfitting_underfitting.md
    │   ├── bias_variance_tradeoff.md
    │   └── overfitting_prevention.md
    ├── optimization/
    │   ├── 1_mse_logistic_regression_convex.md
    │   ├── 2_mse_formula.md
    │   └── ...
    ├── generative_models/
    │   └── generative_vs_discriminative.md
    └── ...
```

可选的配图与代码：
```
topics/{topic}/images/                 # 该主题下的配图（推荐）
topics/{topic}/coding_challenges/      # 该主题相关的代码/挑战（可选）
```

全局共享图片也可放在根目录 `images/` 下（少量通用配图）。

### 2. 命名规范

#### 文档命名
- 全小写 + 下划线或数字前缀，表达清晰、顺序明确
- 单题单文件，文件名体现问题要点

示例：
```
overfitting_underfitting.md
1_mse_logistic_regression_convex.md
generative_vs_discriminative.md
```

#### 图片命名
- `{主题或问题名}_{用途}.png`，如：
```
overfitting_underfitting_analysis.png
softmax_crossentropy_geometry.png
```

### 3. 单文件内容结构（每个问题一个 .md）

建议包含以下部分（强制双语要求）：
```
# {问题标题}

## English Interview Answer
- Clear, concise, directly answer the question
- Include equations/assumptions as needed

## 中文知识点解释（含英文术语标注）
- 用中文阐述关键概念与直觉（intuition）
- 关键术语附英文：如“交叉熵 (Cross-Entropy)”、“相对熵/散度 (KL Divergence)”

## 详细推导/解释
- 数学公式与要点
- 直觉理解（intuition）

## 代码示例（可选）

## 可视化/配图（可选）
![描述](images/{descriptive_name}.png)

## 面试要点（Bullet）
```

说明：
- “English Interview Answer”为正式面试作答版本，要求逻辑清晰、要点明确。
- “中文知识点解释”用于加深理解，关键名词需标注英文原词，便于双语切换与检索。

注意：图片相对路径优先指向 `topics/{topic}/images/`。

### 4. 工作流规范

#### A. 机器学习/AI 题库工作流（topics）
1. 选择或创建主题目录：`topics/{topic}/`
2. 每个问题创建一个 `*.md` 文件；必要时在 `images/` 放入图示
3. 完成后，在根 `README.md` 的相应章节加入该文件的超链接
4. 如涉及示例代码，可在该主题下新建 `coding_challenges/`

#### B. Coding Practice（Notebook）
1. 在 `ml_coding_practice/` 下为每个练习主题新建 `*.ipynb`
2. Notebook 要包含：题目说明、思路要点、从零实现、测试用例、复杂度讨论
3. 如与某个 `topics/{topic}` 强相关，可在 README 中相互链接

#### C. LC.ipynb（Python 操作速查）
1. `LC.ipynb` 专门用于记录“不熟悉的 Python 用法/函数”
2. 我给出一个函数名/用法，你在该 Notebook 中添加：
   - 场景描述（什么时候用）
   - 最小可复现实例（输入/输出）
   - 注意事项/坑点
3. 按模块加目录（如：内置函数、itertools、collections、numpy、pandas等）

### 5. 图片管理规范

#### 引用与生成
```
![图片描述](images/{descriptive_name}.png)
```

生成示例：
```
plt.savefig('topics/{topic}/images/{descriptive_name}.png', dpi=300, bbox_inches='tight')
```

### 6. 代码与注释规范
- 遵循 PEP 8
- 注释用英文；解释必要的“为什么”（rationale）
- 给出最小可运行示例；避免无用样板

### 7. 文档规范
- 根 `README.md` 维护主题导航与超链接
- 每个主题下的文档自包含（读者只看该文件即可理解问题）

## 🎯 开发流程（一览）

### 新增一个 ML/AI 问题
1. 在 `topics/{topic}/` 下新建 `question.md`
2. 补充图示至 `topics/{topic}/images/`（如需要）
3. 在根 `README.md` 的对应主题加入超链接
4. `git add` → `git commit` → `git push`

### 新增一个 Coding Practice 练习
1. 在 `ml_coding_practice/` 下新建 `your_topic.ipynb`
2. 包含从零实现与测试；在 `README.md` 添加入口链接
3. `git add` → `git commit` → `git push`

### 补充 LC.ipynb 用法示例
1. 在 `LC.ipynb` 新增一个函数/用法的条目
2. 编写最小可运行例子与注意事项
3. `git add` → `git commit` → `git push`

## 📝 检查清单

### 添加新内容时：
- [ ] 放在正确的 `topics/{topic}/` 或 `ml_coding_practice/`
- [ ] 使用清晰的文件/图片命名
- [ ] 文档结构完整（概要/推导/要点/可选图与代码）
- [ ] README.md 已添加/更新超链接
- [ ] 代码/Notebook 可运行

### 维护时：
- [ ] 定期检查链接有效性
- [ ] 统一风格与命名
- [ ] 移除或合并冗余内容

## 🔧 工具与质量
- 版本控制：Git（小步提交，信息清晰）
- 代码检查：flake8 / ruff（可选）
- 绘图：Matplotlib / Seaborn / Plotly
- Notebook 规范：分节清晰、单元可运行、结果可复现

## 📚 参考
- PEP 8: https://pep8.org/
- Markdown: https://www.markdownguide.org/
- Matplotlib: https://matplotlib.org/

遵循以上规范可以确保：
1. **一致性**：统一的 topics-first 结构
2. **可维护性**：主题自包含，导航清晰
3. **可扩展性**：便于新增主题与练习
4. **效率**：标准化流程，连接 README，快速同步 GitHub
