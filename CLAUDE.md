# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**agentic-agents** 是一个基于 OpenAI/Claude SDK 的多智能体协作框架，支持技能加载、Agent 间转移控制和会话管理。Python >= 3.11。

## 开发命令

```bash
# 安装依赖（包含开发依赖）
uv pip install -e ".[dev]"

# 运行测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_agent.py -v
```

## 架构概览

```
agentic-agents/
└── src/agentic_agents/    # 主包
    ├── llm/               # LLM 抽象层，OpenAI/Claude 统一接口
    ├── agents/            # Agent 核心实现
    ├── tools/             # 工具注册与基础工具
    ├── skills/            # 技能加载模块（动态从目录加载）
    ├── agent_factory.py   # Agent 工厂函数
    └── session.py         # 会话管理，编排多 Agent 协作
```

### 各层职责

- **llm/**: `OpenAIClient` 和 `AnthropicClient` 实现 `LLMClient` ABC，统一消息格式（`AgentMessage`）和工具调用格式（`ToolCall`）
- **tools/**: `AgentTool` 数据类定义工具结构；`registry.py` 提供全局注册表
- **skills/**: `load_skills_from_directory()` 从子目录加载技能，每个技能含 `skill.md`（frontmatter 元数据 + markdown 指令）和 `tools.py`（`AgentTool` 定义）
- **agents/**: `Agent` 类实现异步流式执行循环（ReAct 模式）；`create_transfer_tool()` 创建交接工具
- **session.py**: `AgentSession` 编排多 Agent 协作，检测 transfer 事件并切换活跃 Agent，支持 Redis 持久化

### Agent 执行流程

1. `build_agents()` 读取 `agents.json`（默认）创建所有 Agent，并自动注入 `create_transfer_tool()` 作为交接工具
2. `AgentSession.process_message()` 接收用户消息，调用当前活跃 Agent 的 `astream()` 方法
3. `Agent.astream()` 循环调用 LLM，执行工具，直到输出 `transfer` 事件或结束
4. Session 检测到 transfer 事件后切换活跃 Agent，继续执行
5. 当控制权回到 Router 时，调用 `_compress_context_if_needed()` 压缩历史消息

### 关键设计模式

- **Router 模式**: 每个系统都包含一个 Router Agent 作为入口，理解用户意图后通过 transfer 工具分发任务
- **Transfer 模式**: Agent 通过调用 `transfer_to_<agent_name>` 工具交接控制权；交接时会传递结果摘要
- **Skill 动态加载**: 技能从目录动态加载，`lookup_skill` 工具让 Agent 运行时查询技能说明

### Skill 目录结构

```
skills/
└── <skill_name>/
    ├── skill.md    # frontmatter: name, description; body: markdown 指令
    └── tools.py    # AgentTool 定义，动态导入
```

### Agent 配置格式 (agents.json)

```json
{
  "metadata": {},
  "agents": [
    {
      "name": "AgentName",
      "duty": "职责描述",
      "skills": ["skill1", "skill2"],
      "system_prompt": "可选，自定义系统提示"
    }
  ]
}
```

### 公共 API

```python
from agentic_agents import (
    Agent, AgentConfig, AgentMeta, create_transfer_tool, AgentState,
    OpenAIClient, AnthropicClient,
    AgentSession, SessionStorage, RedisSessionStorage,
    Skill, load_skills_from_directory,
    build_agent, build_agents,
)
```
