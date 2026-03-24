# agentic-agents

一个基于 LangChain 的多智能体协作框架，支持技能加载、Agent 间转移控制和会话管理。

## 项目结构

```
agentic-agents/
├── agents/          # Agent 核心实现
│   ├── base.py      # Agent 基类
│   ├── handoff.py   # Agent 间转移工具
│   ├── state.py     # 状态定义
│   └── helpers.py   # 辅助函数
├── tools/           # 内置工具
│   └── basic_tools.py
├── skills/          # 技能加载模块
│   ├── loader.py    # 从目录加载技能
│   └── meta_tools.py
├── agent_factory.py # Agent 工厂函数
└── session.py       # 会话管理
```

## 快速开始

```python
from agentic_agents import build_agents

agents = await build_agents(
    tenant_id="tenant_1",
    session_id="session_1",
    skills_dir="./skills",
    runnable_config=None,
)
```
