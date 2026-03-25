"""MCP 工具注册表。

提供全局的 MCP 服务器连接管理和工具注册功能。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_agents.mcp.client import MCPClient
from agentic_agents.mcp.tools import mcp_tools_to_agent_tools
from agentic_agents.mcp.transport import (
    MCPTransport,
    SSEHttpMCPTransport,
    StdioMCPTransport,
    StreamableHttpMCPTransport,
)
from agentic_agents.tools.base import AgentTool

if TYPE_CHECKING:
    from agentic_agents.mcp.transport import StreamableHttpMCPTransport

# 全局客户端和工具注册表
_clients: dict[str, MCPClient] = {}
_tools: dict[str, AgentTool] = {}


async def connect_mcp_server(
    name: str,
    transport: str,
    **kwargs: str | list[str] | dict[str, str],
) -> list[AgentTool]:
    """连接 MCP 服务器并返回可用的工具列表。

    Args:
        name: 服务器名称，用于标识和后续断开连接
        transport: 传输协议类型，"stdio" 或 "http"
        **kwargs: 传输协议相关的参数:
            - stdio: command (str), args (list[str]), env (dict), cwd (str)
            - http: url (str), auth (str), headers (dict)

    Returns:
        可用的 AgentTool 列表

    Raises:
        ValueError: 无效的传输协议

    Example:
        # stdio 传输
        tools = await connect_mcp_server(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )

        # HTTP 传输
        tools = await connect_mcp_server(
            name="remote",
            transport="http",
            url="https://mcp.example.com/sse",
            auth="Bearer token123"
        )
    """
    if name in _clients:
        raise ValueError(f"MCP server '{name}' is already connected")

    # 创建传输层
    if transport == "stdio":
        t = StdioMCPTransport(
            command=str(kwargs["command"]),
            args=kwargs.get("args"),
            env=kwargs.get("env"),
            cwd=kwargs.get("cwd"),
        )
    elif transport in ("http", "sse"):
        t = StreamableHttpMCPTransport(
            url=str(kwargs["url"]),
            auth=kwargs.get("auth"),
            headers=kwargs.get("headers"),
        )
    else:
        raise ValueError(f"Invalid transport: {transport}. Must be 'stdio' or 'http'")

    # 创建客户端并连接
    client = MCPClient(name=name, transport=t)
    await client.connect()

    # 获取工具列表
    mcp_tools = await client.list_tools()
    agent_tools = mcp_tools_to_agent_tools(mcp_tools, client)

    # 注册
    _clients[name] = client
    for tool in agent_tools:
        _tools[tool.name] = tool

    return agent_tools


async def disconnect_mcp_server(name: str) -> None:
    """断开 MCP 服务器连接。

    Args:
        name: 服务器名称

    Raises:
        ValueError: 服务器未连接
    """
    if name not in _clients:
        raise ValueError(f"MCP server '{name}' is not connected")

    client = _clients[name]
    await client.disconnect()
    del _clients[name]

    # 移除该服务器的工具
    global _tools
    _tools = {
        name: tool
        for name, tool in _tools.items()
        if "mcp" not in tool.tags or not any(tag == name for tag in tool.tags)
    }


def get_mcp_tool(name: str) -> AgentTool | None:
    """获取指定的 MCP 工具。

    Args:
        name: 工具名称

    Returns:
        工具实例，如果不存在则返回 None
    """
    return _tools.get(name)


def list_mcp_tools() -> list[AgentTool]:
    """列出所有已注册的 MCP 工具。

    Returns:
        MCP 工具列表
    """
    return list(_tools.values())


def get_mcp_client(name: str) -> MCPClient | None:
    """获取指定的 MCP 客户端。

    Args:
        name: 客户端名称

    Returns:
        客户端实例，如果不存在则返回 None
    """
    return _clients.get(name)


def list_mcp_clients() -> list[MCPClient]:
    """列出所有已连接的 MCP 客户端。

    Returns:
        MCP 客户端列表
    """
    return list(_clients.values())


def clear_mcp_registry() -> None:
    """清空 MCP 注册表，断开所有连接。"""
    global _clients, _tools
    for client in _clients.values():
        import asyncio

        asyncio.create_task(client.disconnect())
    _clients = {}
    _tools = {}
