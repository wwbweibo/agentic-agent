# agentic-agents

一个基于 OpenAI/Claude SDK 的多智能体协作框架，支持技能动态加载、Agent 间转移控制和会话管理。Python >= 3.11。

## 核心特性

- **多 Agent 协作**：通过 Router 入口 Agent 统一调度，支持 Agent 之间转移控制权
- **技能动态加载**：从目录动态加载技能（skill.md + tools.py），无需修改代码即可扩展
- **统一 LLM 抽象**：同时支持 OpenAI 和 Anthropic SDK，通过 `LLMClient` 接口统一封装
- **会话持久化**：内置 Redis / SQLite / 本地文件三种存储后端，支持多租户会话管理
- **ReAct 执行循环**：每个 Agent 内部采用 Thought → Action → Observation 模式循环执行
- **零外部 Agent 框架依赖**：直接使用 OpenAI/Anthropic SDK，无 LangChain 等重型依赖

## 架构概览

```
agentic-agents/
└── src/agentic_agents/
    ├── llm/                 # LLM 抽象层，OpenAI / Anthropic 统一接口
    │   ├── base.py          # LLMClient ABC、AgentMessage、ToolCall 数据类
    │   ├── openai_client.py # OpenAI GPT 系列实现
    │   └── anthropic_client.py # Claude 系列实现
    ├── agents/              # Agent 核心实现
    │   ├── base.py          # Agent 类，ReAct 执行循环
    │   ├── handoff.py       # create_transfer_tool()，Agent 间转移工具
    │   ├── agent_meta.py    # AgentConfig、AgentMeta Pydantic 模型
    │   └── state.py          # AgentState 定义
    ├── tools/               # 工具定义与注册
    │   ├── base.py          # AgentTool 数据类、create_tool() 工厂函数
    │   ├── registry.py      # 全局工具注册表
    │   └── basic_tools.py    # 内置工具（current_time 等）
    ├── skills/              # 技能动态加载
    │   ├── loader.py        # load_skills_from_directory()
    │   └── meta_tools.py    # lookup_skill 工具
    ├── agent_factory.py     # build_agent() / build_agents() 工厂函数
    └── session.py           # AgentSession 会话编排 + 多种存储后端
```

## 安装

```bash
# 安装依赖
uv pip install -e .

# 安装开发依赖（pytest 等）
uv pip install -e ".[dev]"
```

## 快速开始

### 1. 定义 Agent 配置

创建 `agents.json`：

```json
{
  "agents": [
    {
      "name": "OrderAgent",
      "duty": "处理订单相关操作",
      "skills": ["order"]
    }
  ]
}
```

### 2. 创建技能

目录结构：

```
skills/
└── order/
    ├── skill.md   # frontmatter 元数据 + markdown 指令
    └── tools.py   # AgentTool 定义
```

**skill.md** 示例：

```markdown
---
name: order
description: 订单管理和查询技能
---
你是一个订单管理助手，可以帮助用户查询订单状态。
```

**tools.py** 示例：

```python
from agentic_agents.tools.base import create_tool

def get_order(order_id: str) -> str:
    return f"订单 {order_id} 状态：已发货"

order_tool = create_tool(
    name="get_order",
    description="根据订单号查询订单状态",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
    func=get_order,
)
```

### 3. 启动会话

```python
import asyncio
from agentic_agents import build_agents, AgentSession, SessionStorage

async def main():
    # 构建所有 Agent
    agents = await build_agents(
        tenant_id="tenant_1",
        session_id="session_1",
        skills_dir="./skills",
    )

    # 创建会话
    session = AgentSession(
        tenant_id="tenant_1",
        session_id="session_1",
        agent_factory=lambda t, s, sk: agents,
    )

    # 处理消息（事件流）
    async for event in session.process_message("帮我查询订单 12345 的状态"):
        print(event)

asyncio.run(main())
```

## 核心概念

### Router 模式

每个系统都有一个 Router Agent 作为入口。用户消息首先到达 Router，Router 理解意图后通过 `transfer_to_<agent>` 工具将控制权转移给最合适的专家 Agent。

### 转移（Transfer）机制

Expert Agent 通过 `transfer_to_Router` 工具交还控制权时，会传递任务结果摘要。Router 收到后决定是直接回复用户还是继续分发给其他 Agent。

```
用户 → Router → transfer_to_OrderAgent → OrderAgent
                                          ↓
                                   transfer_to_Router
                                          ↓
                                      Router → 用户
```

### 技能（Skill）

技能是可插拔的能力单元，每个技能包含：
- `skill.md`：frontmatter 元数据（name/description）和 markdown 指令
- `tools.py`：一个或多个 `AgentTool` 定义

Agent 启动时自动加载技能目录下的所有技能，并通过 `lookup_skill` 工具动态查询技能说明。

### 会话存储

内置三种存储后端：

| 存储类 | 说明 |
|--------|------|
| `SessionStorage` | 内存存储，默认 |
| `LocalFileSessionStorage` | JSON 文件持久化 |
| `SQLiteSessionStorage` | SQLite 数据库持久化 |
| `RedisSessionStorage` | Redis 缓存持久化 |

均实现 `save_messages`、`load_messages`、`save_response`、`load_response` 异步接口。

## API 参考

### AgentSession

```python
class AgentSession:
    def __init__(
        self,
        tenant_id: str,
        session_id: str,
        agent_factory: Callable[..., Awaitable[dict[str, Agent]]],
        skill_dir: str = "./skills",
        storage: SessionStorage | None = None,
        entry_agent: str = "Router",
        max_epochs: int = 10,
        error_callback: Callable | None = None,
        success_callback: Callable | None = None,
    ) -> None:
        ...

    async def process_message(
        self, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """处理用户消息，yield 事件流。事件类型包括：
        - resp_type="status"：Agent 状态
        - resp_type="text"：文本输出
        - resp_type="tool_result"：工具执行结果
        - resp_type="transfer"：Agent 转移
        - resp_type="finished"：处理完成
        - resp_type="error"：错误
        """
```

### build_agents

```python
async def build_agents(
    tenant_id: str,
    session_id: str,
    skills_dir: str,
    llm: LLMClient | None = None,
    agent_config_path: str = "agents.json",
) -> dict[str, Agent]:
    """构建所有 Agent，返回 name -> Agent 实例的字典。"""
```

### Agent

```python
class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLMClient,
        tools: list[AgentTool],
        system_prompt: str,
        max_epochs: int = 50,
    ) -> None:
        ...

    async def astream(self, state: dict) -> AsyncGenerator[dict, None]:
        """ReAct 执行循环，yield 事件流（text / tool_result / transfer）。"""

    async def invoke(self, state: dict) -> dict:
        """同步调用接口，收集所有事件并返回。"""
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_BASE_URL` | OpenAI API Base URL | - |
| `OPENAI_MODEL` | 模型名称 | `gpt-4o` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | - |

## 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_session_storage.py -v

# 运行单个测试
uv run pytest tests/test_session_integration.py::TestAgentSessionBasic::test_success_callback -v
```

当前测试覆盖：
- `test_session_storage.py`：26 个单元测试，覆盖所有存储后端
- `test_session_integration.py`：15 个集成测试，覆盖 AgentSession 生命周期、转移流程、上下文压缩逻辑
