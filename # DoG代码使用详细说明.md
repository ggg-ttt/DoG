# DoG代码使用详细说明

## 🔍 代码文件详解

### 一、主执行文件使用

#### 1. MetaQA系列 (`KBQA_TASK/metaqa/main_metaqa_*.py`)

**基本结构：**
```python
# 导入必要的库和函数
from metaqa_func import query_metaqa, process_question
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

# 加载数据集并处理
# 调用多智能体辩论框架
# 输出结果
```

**参数说明：**
- `--task`: 指定agentverse中的任务配置路径
- `--output_path`: 指定结果输出文件路径
- `--model`: (可选) 指定使用的模型，如"gpt-4"、"gpt-3.5-turbo"
- `--temperature`: (可选) 控制生成文本的随机性，默认0.7

**使用示例：**
```bash
# 基础使用
python main_metaqa_1hop.py \
    --task "kgqa/metaqa/three_role_one_turn_sequential_metaqa" \
    --output_path "./output/result.txt"

# 指定模型和参数
python main_metaqa_1hop.py \
    --task "kgqa/metaqa/three_role_one_turn_sequential_metaqa" \
    --output_path "./output/result.txt" \
    --model "gpt-4" \
    --temperature 0.5
```

#### 2. Freebase系列 (`KBQA_TASK/freebase/main_*.py`)

**核心功能：**
- 读取JSON格式的数据集
- 调用Freebase API进行实体链接
- 生成和执行SPARQL查询
- 多智能体协作推理

**代码示例：**
```python
# main_cwq.py 中的典型代码片段

import json
from freebase_func import entity_linking, generate_sparql
from agentverse import MultiAgentDebate

# 1. 加载数据
with open('dataset/cwq.json', 'r') as f:
    dataset = json.load(f)

# 2. 对每个问题进行处理
for item in dataset:
    question = item['question']
    
    # 3. 实体链接
    entities = entity_linking(question)
    
    # 4. 启动多智能体辩论
    debate_result = MultiAgentDebate(question, entities)
    
    # 5. 生成SPARQL并查询
    sparql = generate_sparql(debate_result)
    answer = execute_query(sparql)
    
    # 6. 保存结果
    save_result(question, answer, output_path)
```

### 二、功能函数文件使用

#### 1. `freebase_func.py` - Freebase功能函数

**主要函数：**

```python
def entity_linking(question: str) -> List[Dict]:
    """
    实体链接函数
    参数：
        question: 自然语言问题
    返回：
        实体列表，每个实体包含ID、名称、得分等信息
    """
    pass

def get_relations(entity_id: str) -> List[str]:
    """
    获取实体的所有关系
    参数：
        entity_id: Freebase实体ID
    返回：
        关系列表
    """
    pass

def generate_sparql(entities: List, relations: List) -> str:
    """
    生成SPARQL查询
    参数：
        entities: 实体列表
        relations: 关系列表
    返回：
        SPARQL查询字符串
    """
    pass

def execute_query(sparql: str) -> List:
    """
    执行SPARQL查询
    参数：
        sparql: SPARQL查询字符串
    返回：
        查询结果列表
    """
    pass
```

**使用示例：**
```python
from freebase_func import *

# 问题
question = "Who is the president of the United States?"

# 步骤1: 实体链接
entities = entity_linking(question)
print(f"识别到的实体: {entities}")

# 步骤2: 获取关系
for entity in entities:
    relations = get_relations(entity['id'])
    print(f"{entity['name']}的关系: {relations}")

# 步骤3: 生成并执行查询
sparql = generate_sparql(entities, relations)
result = execute_query(sparql)
print(f"答案: {result}")
```

#### 2. `metaqa_func.py` - MetaQA功能函数

**主要函数：**

```python
def load_kb(kb_path: str) -> Dict:
    """加载MetaQA知识库"""
    pass

def find_entity(entity_name: str, kb: Dict) -> str:
    """在知识库中查找实体"""
    pass

def hop_query(entity: str, relation: str, kb: Dict) -> List:
    """执行多跳查询"""
    pass
```

#### 3. `prompt_list.py` - 提示词模板

**结构说明：**
```python
# 问题分解提示
QUESTION_DECOMPOSE_PROMPT = """
你是一个问题分解专家。请将复杂问题分解为简单的子问题。

问题: {question}

请按以下格式输出：
1. 子问题1
2. 子问题2
...
"""

# 实体识别提示
ENTITY_RECOGNITION_PROMPT = """
请识别以下问题中的关键实体：

问题: {question}

输出格式：
- 实体1: [实体类型]
- 实体2: [实体类型]
"""

# 关系预测提示
RELATION_PREDICTION_PROMPT = """
基于以下实体，预测可能的知识图谱关系：

实体: {entities}
问题: {question}

可能的关系：
"""
```

**自定义提示词：**
```python
# 在prompt_list.py中添加新的提示词
CUSTOM_PROMPT = """
你的自定义提示词内容
变量: {variable1}, {variable2}
"""

# 在主程序中使用
from prompt_list import CUSTOM_PROMPT

formatted_prompt = CUSTOM_PROMPT.format(
    variable1="value1",
    variable2="value2"
)
```

### 三、配置文件使用

#### YAML配置文件说明

位置: `agentverse/tasks/kgqa/freebase/three_role_one_turn_sequential_freebase/`

**配置示例：**
```yaml
# config.yaml
agents:
  - name: "Proposer"
    role: "提出答案候选"
    prompt_template: "proposer_prompt"
    
  - name: "Critic"
    role: "批评和质疑"
    prompt_template: "critic_prompt"
    
  - name: "Summarizer"
    role: "总结和决策"
    prompt_template: "summarizer_prompt"

debate:
  max_turns: 3
  consensus_threshold: 0.8
  
model:
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
```

**修改配置：**
1. 打开对应的YAML文件
2. 修改智能体数量、角色或提示词
3. 调整辩论参数
4. 保存并重新运行程序

### 四、评估工具使用

#### `get_evaluation.py` 使用方法

```bash
# 基本使用
python get_evaluation.py \
    --result_file "../KBQA_TASK/freebase/output/cwq_output.txt" \
    --ground_truth "../KBQA_TASK/freebase/dataset/cwq.json"

# 指定评估指标
python get_evaluation.py \
    --result_file "output.txt" \
    --ground_truth "ground_truth.json" \
    --metrics "accuracy,f1,hits@1"

# 输出详细报告
python get_evaluation.py \
    --result_file "output.txt" \
    --ground_truth "ground_truth.json" \
    --detailed_report \
    --output_report "evaluation_report.json"
```

## 🔧 高级用法

### 1. 批量处理

```python
# batch_process.py
import os
import subprocess

datasets = ['cwq', 'grailqa', 'webqsp', 'webquestions']

for dataset in datasets:
    cmd = f"""
    python main_{dataset}.py \
        --task "kgqa/freebase/three_role_one_turn_sequential_freebase" \
        --output_path "./output/{dataset}_output.txt"
    """
    subprocess.run(cmd, shell=True)
```

### 2. 集成到自己的代码

```python
# your_code.py
import sys
sys.path.append('KBQA_TASK/freebase')

from freebase_func import entity_linking, generate_sparql, execute_query

def my_kgqa_pipeline(question):
    # 使用DoG的函数
    entities = entity_linking(question)
    sparql = generate_sparql(entities, [])
    answer = execute_query(sparql)
    return answer

# 使用
result = my_kgqa_pipeline("What is the capital of France?")
print(result)
```

### 3. 自定义智能体

```python
# custom_agent.py
from agentverse.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, name, role):
        super().__init__(name, role)
    
    def generate_response(self, context):
        # 自定义响应逻辑
        prompt = self.build_prompt(context)
        response = self.call_llm(prompt)
        return response
    
    def build_prompt(self, context):
        # 自定义提示词构建
        return f"基于以下上下文生成回答: {context}"
```

## 📊 输出格式说明

### 标准输出格式

```json
{
  "question": "原始问题",
  "entities": ["实体1", "实体2"],
  "relations": ["关系1", "关系2"],
  "sparql": "生成的SPARQL查询",
  "answer": ["答案1", "答案2"],
  "confidence": 0.95,
  "debate_history": [
    {
      "round": 1,
      "agent": "Proposer",
      "content": "提议内容"
    }
  ]
}
```

## 🐛 调试技巧

### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 单步调试

```python
# 在关键位置添加断点
import pdb; pdb.set_trace()
```

### 3. 打印中间结果

```python
print(f"实体链接结果: {entities}")
print(f"生成的SPARQL: {sparql}")
print(f"查询结果: {result}")
```

## 📝 最佳实践

1. **先在小数据集上测试**：使用少量样本验证流程
2. **监控API使用**：注意API调用次数和成本
3. **保存中间结果**：便于调试和分析
4. **版本控制**：记录不同配置的效果
5. **错误处理**：添加try-except捕获异常

## 🆘 获取帮助

如有问题，请：
1. 查看代码注释
2. 阅读原论文
3. 提交GitHub Issue
4. 参考相关项目文档
