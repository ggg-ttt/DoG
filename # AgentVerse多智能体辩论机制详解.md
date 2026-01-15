# AgentVerse多智能体辩论机制详解

## 📚 AgentVerse 文件夹结构

根据项目实际结构，`agentverse`文件夹的核心组织如下：

```
agentverse/
├── __init__.py              # 模块入口，暴露核心接口
├── agents/                  # 智能体定义
│   ├── __init__.py
│   ├── base.py              # 智能体基类
│   ├── agent.py             # 通用智能体实现
│   └── llm.py               # LLM调用封装
├── environments/            # 辩论环境
│   ├── __init__.py
│   ├── base.py              # 环境基类
│   └── rules/               # 辩论规则
│       ├── base.py
│       ├── order/           # 发言顺序规则
│       └── visibility/      # 消息可见性规则
├── message.py               # 消息定义
├── initialization.py        # 系统初始化
├── simulation.py            # 模拟运行器
└── tasks/                   # 任务配置
    └── kgqa/
        ├── freebase/
        │   └── three_role_one_turn_sequential_freebase/
        │       └── config.yaml
        └── metaqa/
            └── three_role_one_turn_sequential_metaqa/
                └── config.yaml
```

---

## 🔑 核心实现机制

### 一、模块入口 (`__init__.py`)

这是整个框架的入口点，负责暴露核心类和函数。

```python
# filepath: agentverse/__init__.py (核心逻辑)

from agentverse.agents import Agent
from agentverse.environments import Environment  
from agentverse.simulation import Simulation
from agentverse.initialization import load_agent, load_environment

# 提供便捷的任务加载函数
def load_task(task_name: str):
    """
    加载指定任务的配置
    
    参数:
        task_name: 任务路径，如 "kgqa/freebase/three_role_one_turn_sequential_freebase"
    
    返回:
        配置好的Simulation对象
    """
    config_path = f"agentverse/tasks/{task_name}/config.yaml"
    return Simulation.from_config(config_path)
```

**使用方式：**
```python
from agentverse import load_task

# 加载KGQA辩论任务
simulation = load_task("kgqa/freebase/three_role_one_turn_sequential_freebase")
result = simulation.run(question="Who directed Inception?")
```

---

### 二、智能体模块 (`agents/`)

#### 2.1 智能体基类 (`agents/base.py`)

定义所有智能体的通用接口和基本行为。

```python
# filepath: agentverse/agents/base.py (核心逻辑)

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from agentverse.message import Message

class BaseAgent(ABC):
    """智能体抽象基类"""
    
    def __init__(
        self,
        name: str,
        role_description: str,
        memory: Optional[List[Message]] = None,
        **kwargs
    ):
        self.name = name
        self.role_description = role_description
        self.memory = memory or []
        
    @abstractmethod
    async def astep(self, env_description: str) -> Message:
        """
        异步执行一步推理
        
        参数:
            env_description: 当前环境描述（包含其他智能体的发言）
        
        返回:
            智能体生成的消息
        """
        pass
    
    def step(self, env_description: str) -> Message:
        """同步执行一步推理"""
        import asyncio
        return asyncio.run(self.astep(env_description))
    
    def add_message_to_memory(self, message: Message):
        """将消息添加到记忆"""
        self.memory.append(message)
    
    def reset(self):
        """重置智能体状态"""
        self.memory = []
```

#### 2.2 LLM智能体 (`agents/agent.py`)

实现基于大语言模型的智能体。

```python
# filepath: agentverse/agents/agent.py (核心逻辑)

from agentverse.agents.base import BaseAgent
from agentverse.message import Message
from agentverse.llm import LLMClient

class Agent(BaseAgent):
    """基于LLM的智能体实现"""
    
    def __init__(
        self,
        name: str,
        role_description: str,
        system_prompt: str,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        super().__init__(name, role_description, **kwargs)
        self.system_prompt = system_prompt
        self.llm = LLMClient(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def astep(self, env_description: str) -> Message:
        """
        执行一步推理
        
        流程:
        1. 构建完整提示词
        2. 调用LLM生成回复
        3. 解析并返回消息
        """
        # 1. 构建提示词
        prompt = self._build_prompt(env_description)
        
        # 2. 调用LLM
        response = await self.llm.agenerate(prompt)
        
        # 3. 创建消息
        message = Message(
            sender=self.name,
            content=response,
            turn=-1  # 将由环境设置
        )
        
        # 4. 保存到记忆
        self.add_message_to_memory(message)
        
        return message
    
    def _build_prompt(self, env_description: str) -> str:
        """
        构建提示词
        
        结构:
        - 系统提示（角色定义）
        - 历史对话
        - 当前环境描述
        """
        prompt_parts = []
        
        # 系统提示
        prompt_parts.append(f"[System]\n{self.system_prompt}\n")
        
        # 角色描述
        prompt_parts.append(f"[Your Role]\n{self.role_description}\n")
        
        # 历史记忆
        if self.memory:
            prompt_parts.append("[Previous Discussion]")
            for msg in self.memory[-10:]:  # 保留最近10条
                prompt_parts.append(f"{msg.sender}: {msg.content}")
        
        # 当前环境
        prompt_parts.append(f"\n[Current Situation]\n{env_description}")
        
        # 请求回复
        prompt_parts.append(f"\n[Your Response as {self.name}]:")
        
        return "\n".join(prompt_parts)
```

#### 2.3 LLM客户端 (`agents/llm.py`)

封装大语言模型的调用。

```python
# filepath: agentverse/agents/llm.py (核心逻辑)

import os
import openai
from typing import List, Dict

class LLMClient:
    """LLM调用客户端"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 配置API
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("OPENAI_API_BASE"):
            openai.api_base = os.environ.get("OPENAI_API_BASE")
    
    async def agenerate(self, prompt: str) -> str:
        """异步生成回复"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM调用错误: {e}")
            return ""
    
    def generate(self, prompt: str) -> str:
        """同步生成回复"""
        import asyncio
        return asyncio.run(self.agenerate(prompt))
```

---

### 三、消息模块 (`message.py`)

定义智能体间通信的消息格式。

```python
# filepath: agentverse/message.py (核心逻辑)

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class Message:
    """智能体消息"""
    
    sender: str                          # 发送者名称
    content: str                         # 消息内容
    receiver: Optional[str] = None       # 接收者（None=广播）
    turn: int = -1                       # 辩论轮次
    msg_type: str = "text"               # 消息类型
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.sender}] (Turn {self.turn}): {self.content[:100]}..."
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "content": self.content,
            "receiver": self.receiver,
            "turn": self.turn,
            "msg_type": self.msg_type,
            "metadata": self.metadata
        }


@dataclass  
class MessagePool:
    """消息池 - 管理所有辩论消息"""
    
    messages: List[Message] = field(default_factory=list)
    
    def add(self, message: Message):
        """添加消息"""
        self.messages.append(message)
    
    def get_visible_messages(
        self, 
        agent_name: str,
        visibility_rule: str = "all"
    ) -> List[Message]:
        """
        获取对特定智能体可见的消息
        
        可见性规则:
        - "all": 所有消息可见
        - "previous": 只能看到自己发言前的消息
        - "none": 看不到其他人的消息
        """
        if visibility_rule == "all":
            return self.messages.copy()
        elif visibility_rule == "previous":
            # 找到该智能体最后一条消息的位置
            visible = []
            for msg in self.messages:
                if msg.sender == agent_name:
                    break
                visible.append(msg)
            return visible
        else:
            return []
    
    def get_by_turn(self, turn: int) -> List[Message]:
        """获取指定轮次的消息"""
        return [m for m in self.messages if m.turn == turn]
    
    def get_last_n(self, n: int) -> List[Message]:
        """获取最近n条消息"""
        return self.messages[-n:] if n < len(self.messages) else self.messages
```

---

### 四、环境模块 (`environments/`)

#### 4.1 环境基类 (`environments/base.py`)

```python
# filepath: agentverse/environments/base.py (核心逻辑)

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from agentverse.agents import BaseAgent
from agentverse.message import Message, MessagePool

class BaseEnvironment(ABC):
    """辩论环境基类"""
    
    def __init__(
        self,
        agents: List[BaseAgent],
        max_turns: int = 3,
        **kwargs
    ):
        self.agents = agents
        self.max_turns = max_turns
        self.message_pool = MessagePool()
        self.current_turn = 0
    
    @abstractmethod
    async def astep(self) -> List[Message]:
        """异步执行一轮辩论"""
        pass
    
    @abstractmethod
    def get_env_description(self, agent: BaseAgent) -> str:
        """获取对特定智能体的环境描述"""
        pass
    
    def reset(self):
        """重置环境"""
        self.message_pool = MessagePool()
        self.current_turn = 0
        for agent in self.agents:
            agent.reset()
```

#### 4.2 顺序辩论环境

这是DoG项目使用的核心环境，实现"三角色单轮顺序辩论"。

```python
# filepath: agentverse/environments/sequential_debate.py (核心逻辑)

from typing import List
from agentverse.environments.base import BaseEnvironment
from agentverse.agents import BaseAgent
from agentverse.message import Message

class SequentialDebateEnvironment(BaseEnvironment):
    """
    顺序辩论环境
    
    特点:
    - 智能体按固定顺序发言
    - 每个智能体可以看到之前所有发言
    - 支持多轮辩论直到达成共识
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        max_turns: int = 3,
        speaking_order: List[str] = None,
        **kwargs
    ):
        super().__init__(agents, max_turns, **kwargs)
        
        # 设置发言顺序
        if speaking_order:
            self.speaking_order = speaking_order
        else:
            self.speaking_order = [agent.name for agent in agents]
    
    async def astep(self) -> List[Message]:
        """
        执行一轮辩论
        
        流程:
        1. 按顺序让每个智能体发言
        2. 每个智能体可以看到之前的所有发言
        3. 收集本轮所有消息
        """
        turn_messages = []
        
        for agent_name in self.speaking_order:
            # 找到对应的智能体
            agent = self._get_agent_by_name(agent_name)
            if agent is None:
                continue
            
            # 构建环境描述
            env_desc = self.get_env_description(agent)
            
            # 获取智能体回复
            message = await agent.astep(env_desc)
            message.turn = self.current_turn
            
            # 添加到消息池
            self.message_pool.add(message)
            turn_messages.append(message)
        
        self.current_turn += 1
        return turn_messages
    
    def get_env_description(self, agent: BaseAgent) -> str:
        """
        构建环境描述
        
        包含:
        - 当前问题
        - 之前所有智能体的发言
        - 对当前智能体的期望
        """
        desc_parts = []
        
        # 添加之前的发言
        visible_messages = self.message_pool.get_visible_messages(
            agent.name, 
            visibility_rule="all"
        )
        
        if visible_messages:
            desc_parts.append("Previous discussion:")
            for msg in visible_messages:
                desc_parts.append(f"  [{msg.sender}]: {msg.content}")
        
        return "\n".join(desc_parts)
    
    def _get_agent_by_name(self, name: str) -> BaseAgent:
        """根据名称获取智能体"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
```

---

### 五、模拟运行器 (`simulation.py`)

协调整个辩论流程的执行。

```python
# filepath: agentverse/simulation.py (核心逻辑)

import yaml
from typing import Dict, Any, List
from agentverse.agents import Agent
from agentverse.environments import SequentialDebateEnvironment
from agentverse.message import Message

class Simulation:
    """辩论模拟器"""
    
    def __init__(
        self,
        agents: List[Agent],
        environment: SequentialDebateEnvironment,
        max_turns: int = 3
    ):
        self.agents = agents
        self.environment = environment
        self.max_turns = max_turns
    
    @classmethod
    def from_config(cls, config_path: str) -> "Simulation":
        """从配置文件创建模拟器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建智能体
        agents = []
        for agent_config in config['agents']:
            agent = Agent(
                name=agent_config['name'],
                role_description=agent_config['role_description'],
                system_prompt=agent_config.get('system_prompt', ''),
                model_name=agent_config.get('model', 'gpt-4'),
                temperature=agent_config.get('temperature', 0.7)
            )
            agents.append(agent)
        
        # 创建环境
        env_config = config.get('environment', {})
        environment = SequentialDebateEnvironment(
            agents=agents,
            max_turns=env_config.get('max_turns', 3),
            speaking_order=env_config.get('speaking_order')
        )
        
        return cls(
            agents=agents,
            environment=environment,
            max_turns=env_config.get('max_turns', 3)
        )
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        运行完整的辩论流程
        
        参数:
            question: 需要辩论的问题
            
        返回:
            包含最终答案和辩论历史的字典
        """
        import asyncio
        return asyncio.run(self.arun(question))
    
    async def arun(self, question: str) -> Dict[str, Any]:
        """异步运行辩论"""
        
        # 1. 重置环境
        self.environment.reset()
        
        # 2. 添加初始问题到消息池
        initial_message = Message(
            sender="System",
            content=f"Question to debate: {question}",
            turn=0
        )
        self.environment.message_pool.add(initial_message)
        
        # 3. 执行多轮辩论
        all_messages = []
        for turn in range(self.max_turns):
            turn_messages = await self.environment.astep()
            all_messages.extend(turn_messages)
            
            # 检查是否达成共识
            if self._check_consensus(turn_messages):
                break
        
        # 4. 提取最终答案
        final_answer = self._extract_final_answer()
        
        return {
            'question': question,
            'answer': final_answer,
            'total_turns': self.environment.current_turn,
            'debate_history': [msg.to_dict() for msg in all_messages]
        }
    
    def _check_consensus(self, messages: List[Message]) -> bool:
        """
        检查是否达成共识
        
        简单策略：检查Summarizer是否给出了明确答案
        """
        for msg in messages:
            if "Summarizer" in msg.sender:
                # 检查是否包含明确的答案标记
                if "Final Answer:" in msg.content or "最终答案:" in msg.content:
                    return True
        return False
    
    def _extract_final_answer(self) -> str:
        """从辩论历史中提取最终答案"""
        messages = self.environment.message_pool.messages
        
        # 从后往前找Summarizer的发言
        for msg in reversed(messages):
            if "Summarizer" in msg.sender:
                return msg.content
        
        # 如果没有Summarizer，返回最后一条消息
        if messages:
            return messages[-1].content
        
        return "No answer generated"
```

---

### 六、任务配置 (YAML配置文件)

#### 6.1 Freebase任务配置

```yaml
# filepath: agentverse/tasks/kgqa/freebase/three_role_one_turn_sequential_freebase/config.yaml

task_name: "KGQA_Freebase_ThreeRole"
description: "Three-role debate for KGQA on Freebase"

# 智能体配置
agents:
  - name: "Proposer"
    role_description: |
      You are the Answer Proposer in a knowledge graph QA debate.
      Your job is to:
      1. Analyze the given question carefully
      2. Identify relevant entities and relations
      3. Propose candidate answers with reasoning
      
      Always explain your reasoning process step by step.
    system_prompt: |
      You are participating in a multi-agent debate to answer questions 
      using a knowledge graph. Be analytical and thorough.
    model: "gpt-4"
    temperature: 0.7

  - name: "Critic"  
    role_description: |
      You are the Critical Reviewer in a knowledge graph QA debate.
      Your job is to:
      1. Carefully examine the proposed answers
      2. Identify potential issues, errors, or missing information
      3. Challenge weak reasoning and ask clarifying questions
      
      Be constructive but rigorous in your criticism.
    system_prompt: |
      You are a critical thinker. Question assumptions and 
      look for logical flaws.
    model: "gpt-4"
    temperature: 0.8

  - name: "Summarizer"
    role_description: |
      You are the Decision Maker in a knowledge graph QA debate.
      Your job is to:
      1. Consider all arguments from Proposer and Critic
      2. Weigh the evidence and reasoning
      3. Provide the final answer with confidence level
      
      Format your final answer as:
      Final Answer: [your answer]
      Confidence: [high/medium/low]
      Reasoning: [brief explanation]
    system_prompt: |
      You are a fair judge. Synthesize different viewpoints and 
      make balanced decisions.
    model: "gpt-4"
    temperature: 0.6

# 环境配置
environment:
  type: "SequentialDebate"
  max_turns: 1                    # DoG使用单轮辩论
  speaking_order:
    - "Proposer"
    - "Critic"
    - "Summarizer"
  message_visibility: "all"       # 所有消息对所有智能体可见

# 输出配置
output:
  save_debate_history: true
  format: "json"
```

#### 6.2 MetaQA任务配置

```yaml
# filepath: agentverse/tasks/kgqa/metaqa/three_role_one_turn_sequential_metaqa/config.yaml

task_name: "KGQA_MetaQA_ThreeRole"
description: "Three-role debate for KGQA on MetaQA"

agents:
  - name: "Proposer"
    role_description: |
      You are analyzing questions about movies, actors, directors, etc.
      Use your knowledge to propose answers based on the MetaQA knowledge base.
      
      For multi-hop questions, break them down into steps:
      - 1-hop: Direct relation query
      - 2-hop: Two-step relation traversal
      - 3-hop: Three-step relation traversal
    system_prompt: "You are a movie knowledge expert."
    model: "gpt-4"
    temperature: 0.7

  - name: "Critic"
    role_description: |
      Review the proposed answers for movie-related questions.
      Check for:
      - Correct entity identification
      - Valid relation paths
      - Logical consistency
    system_prompt: "You are a critical reviewer of movie knowledge."
    model: "gpt-4"
    temperature: 0.8

  - name: "Summarizer"
    role_description: |
      Synthesize the debate and provide the final answer.
      Format: Final Answer: [answer]
    system_prompt: "You make final decisions based on debate."
    model: "gpt-4"
    temperature: 0.6

environment:
  type: "SequentialDebate"
  max_turns: 1
  speaking_order: ["Proposer", "Critic", "Summarizer"]
```

---

## 🔄 完整工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    KGQA_TASK/main_*.py                      │
│                         调用入口                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              agentverse.load_task(task_name)                │
│                   加载任务配置                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Simulation.from_config()                       │
│    ┌──────────────────────────────────────────────────┐     │
│    │  1. 解析config.yaml                               │     │
│    │  2. 创建Agent实例 (Proposer, Critic, Summarizer) │     │
│    │  3. 创建SequentialDebateEnvironment              │     │
│    └──────────────────────────────────────────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  simulation.run(question)                    │
│    ┌──────────────────────────────────────────────────┐     │
│    │  Turn 1:                                          │     │
│    │  ┌─────────────────────────────────────────────┐ │     │
│    │  │ Proposer.astep() → 提出候选答案              │ │     │
│    │  │      ↓ (消息添加到MessagePool)              │ │     │
│    │  │ Critic.astep()   → 批评和质疑               │ │     │
│    │  │      ↓ (消息添加到MessagePool)              │ │     │
│    │  │ Summarizer.astep() → 总结给出最终答案       │ │     │
│    │  └─────────────────────────────────────────────┘ │     │
│    │                                                   │     │
│    │  检查共识 → 如果达成则结束，否则继续下一轮        │     │
│    └──────────────────────────────────────────────────┘     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     返回结果                                 │
│  {                                                          │
│    "question": "原始问题",                                   │
│    "answer": "最终答案",                                     │
│    "total_turns": 1,                                        │
│    "debate_history": [...]                                  │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 三角色辩论机制详解

### 角色分工

| 角色 | 英文名 | 职责 | 特点 |
|------|--------|------|------|
| 提议者 | Proposer | 分析问题，提出候选答案 | 积极主动，逻辑清晰 |
| 批评者 | Critic | 审查答案，指出问题 | 批判性思维，严谨 |
| 总结者 | Summarizer | 综合意见，最终决策 | 平衡各方，果断 |

### 辩论流程

```
Question: "Who directed the movie Inception?"

Round 1:
┌─────────────────────────────────────────────────────────────┐
│ [Proposer]:                                                 │
│ Let me analyze this question about the movie Inception.     │
│                                                             │
│ Entity identified: "Inception" (movie)                      │
│ Relation needed: "director"                                 │
│                                                             │
│ Candidate Answer: Christopher Nolan                         │
│ Reasoning: Christopher Nolan is well-known for directing    │
│ Inception (2010), which he also wrote.                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ [Critic]:                                                   │
│ I'll review the Proposer's answer.                          │
│                                                             │
│ Strengths:                                                  │
│ - Correct entity identification                             │
│ - Valid reasoning about Christopher Nolan                   │
│                                                             │
│ Potential Issues:                                           │
│ - Should verify this is the only "Inception" movie          │
│ - Confidence seems high, which is appropriate here          │
│                                                             │
│ Assessment: The answer appears correct and well-reasoned.   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ [Summarizer]:                                               │
│ Based on the debate:                                        │
│                                                             │
│ - Proposer identified Christopher Nolan as director         │
│ - Critic validated the reasoning with minor caveats         │
│ - Both agree on the answer                                  │
│                                                             │
│ Final Answer: Christopher Nolan                             │
│ Confidence: High                                            │
│ Reasoning: Clear consensus with valid knowledge graph path  │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 关键设计特点

### 1. 模块化设计
- 智能体、环境、消息各自独立
- 易于扩展和替换组件

### 2. 配置驱动
- 通过YAML配置定义任务
- 无需修改代码即可调整参数

### 3. 异步支持
- 支持异步执行提高效率
- 可并行调用多个LLM

### 4. 灵活的消息可见性
- 可配置智能体看到哪些消息
- 支持不同的辩论策略

### 5. 可扩展的角色系统
- 易于添加新的智能体角色
- 自定义角色行为和提示词

---

## 🔧 如何修改辩论策略

### 1. 添加新角色

在`config.yaml`中添加新的智能体配置：

```yaml
agents:
  # ...existing agents...
  
  - name: "Verifier"
    role_description: |
      You verify answers against the knowledge graph.
      Check if the answer can be reached through valid paths.
    model: "gpt-4"
    temperature: 0.5
```

### 2. 修改发言顺序

```yaml
environment:
  speaking_order:
    - "Proposer"
    - "Critic"
    - "Verifier"    # 新增
    - "Summarizer"
```

### 3. 增加辩论轮数

```yaml
environment:
  max_turns: 3  # 从1轮改为3轮
```

### 4. 修改模型参数

```yaml
agents:
  - name: "Proposer"
    model: "gpt-4-turbo"    # 使用更快的模型
    temperature: 0.5        # 降低随机性
    max_tokens: 3000        # 增加输出长度
```

---

这份文档完整解释了AgentVerse框架实现多智能体辩论的所有核心机制！
