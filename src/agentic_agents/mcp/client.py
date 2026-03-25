"""MCP 客户端封装。

对 MCP SDK 的 ClientSession 进行封装，提供更简洁的 API。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.client.session import ClientSession
from mcp.types import CallToolResult, ListToolsResult, Tool

from agentic_agents.mcp.transport import MCPTransport


@dataclass
class MCPTool:
    """MCP 工具定义。

    属性与 mcp.types.Tool 兼容，但更易于使用。
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    _raw: Tool

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> MCPTool:
        """从 mcp.types.Tool 创建一个 MCPTool。"""
        return cls(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema,
            _raw=tool,
        )


class MCPClient:
    """MCP 客户端封装。

    封装 MCP 会话管理，提供简化的 API。

    Example:
        transport = StdioMCPTransport(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        async with MCPClient(name="filesystem", transport=transport) as client:
            tools = await client.list_tools()
            result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})
    """

    def __init__(
        self,
        name: str,
        transport: MCPTransport,
    ):
        self.name = name
        self.transport = transport
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """建立 MCP 连接并初始化会话。"""
        self._session = None
        # 会话在 context manager 中自动初始化
        context = self.transport.session()
        self._session_context = context
        self._session = await context.__aenter__()

    async def disconnect(self) -> None:
        """断开 MCP 连接。"""
        if hasattr(self, "_session_context") and self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
            self._session = None

    @property
    def session(self) -> ClientSession:
        """获取当前会话。"""
        if self._session is None:
            raise RuntimeError(
                f"MCP client '{self.name}' is not connected. Call connect() first."
            )
        return self._session

    async def list_tools(self) -> list[MCPTool]:
        """列出 MCP 服务器提供的所有工具。"""
        result: ListToolsResult = await self.session.list_tools()
        return [MCPTool.from_mcp_tool(tool) for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用 MCP 工具并返回结果。

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果的文本表示
        """
        if arguments is None:
            arguments = {}

        result: CallToolResult = await self.session.call_tool(name, arguments)

        # 将结果转换为文本
        if not result.content:
            return ""

        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif hasattr(item, "data"):
                parts.append(str(item.data))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    async def __aenter__(self) -> MCPClient:
        """支持 async with 语法。"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """支持 async with 语法。"""
        await self.disconnect()

    def __repr__(self) -> str:
        return f"MCPClient(name={self.name!r}, transport={self.transport!r})"
