# 🛠️ 开发规范文档

## 📋 项目开发规范

### 1. 文件组织结构

#### 问题文件
```
questions/
├── {category}/              # 技术分类
│   └── {topic_name}.md     # 具体问题文件
```

**示例：**
```
questions/
├── basic_ml/
│   ├── overfitting_underfitting.md
│   ├── bias_variance_tradeoff.md
│   └── cross_validation.md
├── deep_learning/
│   ├── neural_networks.md
│   ├── backpropagation.md
│   └── activation_functions.md
```

#### 答案文件
```
answers/
├── {category}/              # 与技术分类对应
│   └── {topic_name}.md     # 与问题文件对应
```

#### 图片文件
```
images/
├── {category}/              # 与技术分类对应
│   ├── {topic_name}_analysis.png
│   ├── {topic_name}_visualization.png
│   └── {topic_name}_architecture.png
```

### 2. 命名规范

#### 文件命名
- 使用小写字母和下划线
- 使用描述性名称
- 保持一致性

**示例：**
```
✅ 正确: overfitting_underfitting.md
✅ 正确: attention_mechanism.md
✅ 正确: gradient_descent_optimization.md

❌ 错误: Overfitting.md
❌ 错误: attention-mechanism.md
❌ 错误: question1.md
```

#### 图片命名
- 使用描述性名称
- 包含分析类型
- 使用.png或.jpg格式

**示例：**
```
✅ 正确: overfitting_underfitting_analysis.png
✅ 正确: attention_weights_visualization.png
✅ 正确: model_architecture_diagram.png
```

### 3. 内容规范

#### 问题文件结构
```markdown
# {技术领域}问题

## 问题1: {问题标题}

{问题描述}

### 背景
{背景信息}

### 要求
{具体要求}

### 难度
{初级/中级/高级}
```

#### 答案文件结构
```markdown
# {技术领域}问题 - 详细答案

## 问题1: {问题标题}

### 中文理解
{中文解释，便于记忆}

### 英文标准面试答案
{英文标准回答}

### 数学原理
{相关数学公式和推导}

### 代码示例
{实际代码演示}

### 可视化
![图片描述](../images/{category}/image_name.png)

### 面试常见问题
{Follow-up问题}

### 关键要点
{总结要点}
```

### 4. 图片管理规范

#### 图片引用格式
```markdown
![图片描述](../images/{category}/image_name.png)
```

#### 图片生成脚本
```python
# 保存到正确目录
plt.savefig(f'images/{category}/{topic_name}_{analysis_type}.png', 
            dpi=300, bbox_inches='tight')
```

### 5. 代码规范

#### Python代码
- 使用PEP 8规范
- 添加详细注释
- 包含错误处理
- 提供使用示例

#### 测试代码
- 包含单元测试
- 提供集成测试
- 性能基准测试
- 清晰的测试报告

### 6. 文档规范

#### README文件
- 清晰的项目介绍
- 详细的使用说明
- 完整的安装指南
- 贡献指南

#### 代码注释
- 函数和类的文档字符串
- 复杂逻辑的行内注释
- 参数和返回值说明
- 使用示例

## 🎯 开发流程

### 1. 添加新问题
1. 在 `questions/{category}/` 创建问题文件
2. 在 `answers/{category}/` 创建对应答案文件
3. 生成相关图片并保存到 `images/{category}/`
4. 在答案文件中引用图片
5. 测试问题生成器功能

### 2. 添加编程挑战
1. 在 `coding_challenges/` 创建新目录
2. 实现核心代码
3. 编写测试用例
4. 创建README说明文档
5. 添加示例和扩展挑战

### 3. 更新Pipeline
1. 修改配置文件
2. 更新相关模块
3. 测试完整流程
4. 更新文档

## 📝 检查清单

### 添加新内容时：
- [ ] 文件放在正确目录
- [ ] 使用规范的文件名
- [ ] 内容结构完整
- [ ] 图片正确引用
- [ ] 代码可以运行
- [ ] 测试通过

### 维护时：
- [ ] 检查链接有效性
- [ ] 更新过时内容
- [ ] 保持格式一致
- [ ] 优化文件大小

## 🔧 工具推荐

### 开发工具
- **编辑器**: VS Code, PyCharm
- **版本控制**: Git
- **文档**: Markdown
- **图表**: Matplotlib, Seaborn, Plotly

### 质量保证
- **代码检查**: flake8, black
- **测试**: pytest
- **文档**: Sphinx
- **图片**: GIMP, Inkscape

## 📚 参考资源

### 技术文档
- [PEP 8 - Python代码规范](https://pep8.org/)
- [Markdown语法](https://www.markdownguide.org/)
- [Matplotlib文档](https://matplotlib.org/)

### 最佳实践
- 保持代码简洁
- 文档及时更新
- 测试覆盖完整
- 图片质量清晰

遵循这些规范可以确保项目的：
1. **一致性**: 所有内容格式统一
2. **可维护性**: 结构清晰，易于维护
3. **可扩展性**: 便于添加新内容
4. **专业性**: 符合行业标准
