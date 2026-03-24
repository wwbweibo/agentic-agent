"""
agentic-agents - 一个基于 OpenAI/Claude SDK 的多智能体协作框架。

主要模块:
- agents: Agent 核心实现
- llm: LLM 抽象层（OpenAI / Anthropic）
- tools: 工具定义和注册
- skills: 技能加载模块
"""

from .agent_factory import build_agent, build_agents
from .agents.base import Agent
from .agents.agent_meta import AgentConfig, AgentMeta
from .agents.handoff import create_transfer_tool
from .agents.state import AgentState
from .llm import AnthropicClient, OpenAIClient
from .session import AgentSession, RedisSessionStorage, SessionStorage
from .skills.loader import Skill, load_skills_from_directory

__all__ = [
    "build_agent",
    "build_agents",
    "Agent",
    "AgentConfig",
    "AgentMeta",
    "create_transfer_tool",
    "AgentState",
    "OpenAIClient",
    "AnthropicClient",
    "AgentSession",
    "SessionStorage",
    "RedisSessionStorage",
    "Skill",
    "load_skills_from_directory",
]
