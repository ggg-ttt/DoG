# DoG项目中文使用指南

## 📋 项目简介

DoG (Debate on Graph) 是一个基于大语言模型的知识图谱问答推理框架。通过多智能体辩论机制，该框架能够有效提升大语言模型在知识图谱上的推理准确性和可靠性。

## 🏗️ 项目架构说明

### 核心模块

#### 1. `agentverse/` - 多智能体框架
- **位置**: `agentverse/tasks/kgqa/`
- **功能**: 定义多个AI智能体的角色、交互方式和辩论策略
- **子模块**:
  - `freebase/`: Freebase知识图谱相关任务配置
  - `metaqa/`: MetaQA知识图谱相关任务配置

#### 2. `KBQA_TASK/` - 知识图谱问答任务
这是项目的主要执行目录，包含所有数据集的处理和推理代码。

##### Freebase相关任务 (`KBQA_TASK/freebase/`)
- **`freebase_func.py`**: Freebase知识图谱的核心功能函数
  - 实体链接
  - 关系检索
  - SPARQL查询生成和执行
  
- **主执行文件**:
  - `main_cwq.py`: 处理Complex WebQuestions数据集
  - `main_grailqa.py`: 处理GrailQA数据集
  - `main_webqsp.py`: 处理WebQSP数据集
  - `main_webquestions.py`: 处理WebQuestions数据集

- **`prompt_list.py`**: 存储所有提示词模板

##### MetaQA相关任务 (`KBQA_TASK/metaqa/`)
- **`metaqa_func.py`**: MetaQA知识图谱的核心功能函数
- **主执行文件**:
  - `main_metaqa_1hop.py`: 1跳问题（简单问题）
  - `main_metaqa_2hop.py`: 2跳问题（中等难度）
  - `main_metaqa_3hop.py`: 3跳问题（复杂问题）

#### 3. `eval_helper/` - 评估工具
- **`get_evaluation.py`**: 计算准确率、F1分数等评估指标

## 🚀 快速开始

### 步骤1: 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 配置API密钥

**方式一：使用环境变量（推荐）**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**方式二：在代码中配置**
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

**使用本地模型（可选）**
```python
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"  # 你的本地模型API地址
```

### 步骤3: 配置知识图谱服务

建议按照[此教程](https://github.com/dki-lab/Freebase-Setup)在本地部署Virtuoso服务器以使用Freebase知识图谱。

## 📝 使用示例

### 运行MetaQA数据集

```bash
# 切换到KBQA_TASK/metaqa目录
cd KBQA_TASK/metaqa

# 运行1跳问题
python main_metaqa_1hop.py \
    --task "kgqa/metaqa/three_role_one_turn_sequential_metaqa" \
    --output_path "./output/metaqa_1hop_output.txt"

# 运行2跳问题
python main_metaqa_2hop.py \
    --task "kgqa/metaqa/three_role_one_turn_sequential_metaqa" \
    --output_path "./output/metaqa_2hop_output.txt"

# 运行3跳问题
python main_metaqa_3hop.py \
    --task "kgqa/metaqa/three_role_one_turn_sequential_metaqa" \
    --output_path "./output/metaqa_3hop_output.txt"
```

### 运行Freebase数据集

```bash
# 切换到KBQA_TASK/freebase目录
cd KBQA_TASK/freebase

# 运行CWQ数据集
python main_cwq.py \
    --task "kgqa/freebase/three_role_one_turn_sequential_freebase" \
    --output_path "./output/cwq_output.txt"

# 运行GrailQA数据集
python main_grailqa.py \
    --task "kgqa/freebase/three_role_one_turn_sequential_freebase" \
    --output_path "./output/grailqa_output.txt"

# 运行WebQSP数据集
python main_webqsp.py \
    --task "kgqa/freebase/three_role_one_turn_sequential_freebase" \
    --output_path "./output/webqsp_output.txt"

# 运行WebQuestions数据集
python main_webquestions.py \
    --task "kgqa/freebase/three_role_one_turn_sequential_freebase" \
    --output_path "./output/webquestions_output.txt"
```

## 🔧 自定义配置

### 修改智能体策略

1. 找到配置文件：`agentverse/tasks/kgqa/freebase/`或`agentverse/tasks/kgqa/metaqa/`
2. 编辑YAML文件来修改：
   - 智能体角色
   - 辩论轮数
   - 问题简化策略
   - 提示词模板

### 修改提示词

编辑对应的`prompt_list.py`文件，可以自定义：
- 问题分解提示
- 实体识别提示
- 关系预测提示
- 答案验证提示

## 📊 查看结果

执行完成后，结果会保存在指定的输出文件中（如`./output/cwq_output.txt`）。

使用评估工具查看性能指标：

```bash
cd eval_helper
python get_evaluation.py --result_file "../KBQA_TASK/freebase/output/cwq_output.txt"
```

## 🎯 数据集说明

| 数据集 | 知识图谱 | 难度 | 问题数量 | 位置 |
|--------|----------|------|----------|------|
| MetaQA-1hop | MetaQA | 简单 | - | `KBQA_TASK/metaqa/dataset/` |
| MetaQA-2hop | MetaQA | 中等 | - | `KBQA_TASK/metaqa/dataset/` |
| MetaQA-3hop | MetaQA | 困难 | - | `KBQA_TASK/metaqa/dataset/` |
| WebQuestions | Freebase | 简单 | - | `KBQA_TASK/freebase/dataset/WebQuestions.json` |
| WebQSP | Freebase | 中等 | - | `KBQA_TASK/freebase/dataset/WebQSP.json` |
| CWQ | Freebase | 困难 | - | `KBQA_TASK/freebase/dataset/cwq.json` |
| GrailQA | Freebase | 困难 | - | `KBQA_TASK/freebase/dataset/grailqa.json` |

## 💡 工作流程

1. **问题输入**: 读取数据集中的自然语言问题
2. **问题简化**: 通过多智能体辩论简化复杂问题
3. **实体链接**: 识别问题中的实体并链接到知识图谱
4. **关系预测**: 预测问题涉及的知识图谱关系
5. **查询生成**: 生成SPARQL查询语句
6. **答案获取**: 执行查询并返回答案
7. **答案验证**: 通过辩论机制验证答案的正确性

## ⚠️ 常见问题

### 1. API调用失败
- 检查API密钥是否正确设置
- 确认网络连接正常
- 检查API额度是否充足

### 2. 知识图谱连接失败
- 确认Virtuoso服务器已启动
- 检查连接配置是否正确
- 验证知识图谱数据是否完整

### 3. 内存不足
- 减少批处理大小
- 使用更小的模型
- 增加系统内存

## 📚 参考资料

- [原论文](https://arxiv.org/abs/your-paper-link)
- [Freebase设置教程](https://github.com/dki-lab/Freebase-Setup)
- [ChatEval项目](https://github.com/thunlp/ChatEval)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

请参考项目根目录的LICENSE文件。
