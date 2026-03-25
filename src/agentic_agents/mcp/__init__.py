"""MCP (Model Context Protocol) 支持模块。

提供 MCP 服务器连接和工具管理功能，支持 stdio 和 streamable-http 两种传输协议。

Example:
    # stdio 传输 - 连接本地 MCP 服务器
    from agentic_agents.mcp import connect_mcp_server

    tools = await connect_mcp_server(
        name="filesystem",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    # 使用工具
    from agentic_agents import build_agent
    agent = build_agent("my_agent", tools=tools)

    # HTTP 传输 - 连接远程 MCP 服务器
    tools = await connect_mcp_server(
        name="remote",
        transport="http",
        url="https://mcp.example.com/sse",
        auth="Bearer token123"
    )
"""
from agentic_agents.mcp.client import MCPClient, MCPTool
from agentic_agents.mcp.registry import (
    clear_mcp_registry,
    connect_mcp_server,
    disconnect_mcp_server,
    get_mcp_client,
    get_mcp_tool,
    list_mcp_clients,
    list_mcp_tools,
)
from agentic_agents.mcp.tools import MCPToolAdapter, mcp_tools_to_agent_tools
from agentic_agents.mcp.transport import (
    MCPTransport,
    SSEHttpMCPTransport,
    StdioMCPTransport,
    StreamableHttpMCPTransport,
)

__all__ = [
    # 传输层
    "MCPTransport",
    "StdioMCPTransport",
    "StreamableHttpMCPTransport",
    "SSEHttpMCPTransport",
    # 客户端
    "MCPClient",
    "MCPTool",
    # 工具适配
    "MCPToolAdapter",
    "mcp_tools_to_agent_tools",
    # 注册表
    "connect_mcp_server",
    "disconnect_mcp_server",
    "get_mcp_client",
    "get_mcp_tool",
    "list_mcp_clients",
    "list_mcp_tools",
    "clear_mcp_registry",
]
