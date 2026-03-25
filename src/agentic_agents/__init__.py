"""
agentic-agents - 一个基于 OpenAI/Claude SDK 的多智能体协作框架。

主要模块:
- agents: Agent 核心实现
- llm: LLM 抽象层（OpenAI / Anthropic）
- tools: 工具定义和注册
- skills: 技能加载模块
- mcp: MCP (Model Context Protocol) 支持
"""

from .agent_factory import build_agent, build_agents
from .agents.agent_meta import AgentConfig, AgentMeta, MCPServerConfig
from .agents.base import Agent
from .agents.handoff import create_transfer_tool
from .agents.state import AgentState
from .llm import AnthropicClient, OpenAIClient
from .mcp import (
    MCPClient,
    StdioMCPTransport,
    StreamableHttpMCPTransport,
    connect_mcp_server,
    disconnect_mcp_server,
    get_mcp_tool,
    list_mcp_tools,
)
from .session import AgentSession, RedisSessionStorage, SessionStorage
from .skills.loader import Skill, load_skills_from_directory

__all__ = [
    # Agent 工厂
    "build_agent",
    "build_agents",
    # Agent 配置
    "Agent",
    "AgentConfig",
    "AgentMeta",
    "MCPServerConfig",
    "create_transfer_tool",
    "AgentState",
    # LLM
    "OpenAIClient",
    "AnthropicClient",
    # Session
    "AgentSession",
    "SessionStorage",
    "RedisSessionStorage",
    # Skills
    "Skill",
    "load_skills_from_directory",
    # MCP
    "MCPClient",
    "StdioMCPTransport",
    "StreamableHttpMCPTransport",
    "connect_mcp_server",
    "disconnect_mcp_server",
    "get_mcp_tool",
    "list_mcp_tools",
]
