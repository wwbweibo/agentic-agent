"""MCP 传输层实现 - 支持 stdio 和 streamable-http 两种协议。

基于官方 mcp Python SDK，实现传输层适配。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

import anyio
import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPTransport(ABC):
    """MCP 传输基类。"""

    @abstractmethod
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """创建一个 MCP ClientSession 会话。

        返回的会话已初始化，可以直接调用 list_tools()、call_tool() 等方法。
        """
        pass


class StdioMCPTransport(MCPTransport):
    """通过子进程 stdio 与 MCP 服务器通信。

    适合本地 MCP 服务器，如通过 npx 启动的 Node.js 服务器。

    Example:
        transport = StdioMCPTransport(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        async with transport.session() as session:
            tools = await session.list_tools()
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._server_params = StdioServerParameters(
            command=command,
            args=self.args,
            env=env,
            cwd=cwd,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """创建 stdio 会话。"""
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                yield session

    def __repr__(self) -> str:
        return f"StdioMCPTransport(command={self.command!r}, args={self.args})"


class StreamableHttpMCPTransport(MCPTransport):
    """通过 HTTP streamable 协议与 MCP 服务器通信。

    适合远程 MCP 服务器，支持 Bearer Token 和 Basic 认证。

    Example:
        transport = StreamableHttpMCPTransport(
            url="https://mcp.example.com/sse",
            auth="Bearer token123"
        )
        async with transport.session() as session:
            tools = await session.list_tools()

        # 或使用 Basic 认证
        import base64
        credentials = base64.b64encode(b"user:pass").decode()
        transport = StreamableHttpMCPTransport(
            url="https://mcp.example.com/sse",
            auth=f"Basic {credentials}"
        )
    """

    def __init__(
        self,
        url: str,
        auth: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.auth = auth
        self.headers = headers or {}

    def _create_http_client(self) -> httpx.AsyncClient:
        """创建配置好认证的 HTTP 客户端。"""
        client_headers = dict(self.headers)

        if self.auth:
            client_headers["Authorization"] = self.auth

        return httpx.AsyncClient(
            headers=client_headers,
            follow_redirects=True,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """创建 streamable-http 会话。"""
        http_client = self._create_http_client()
        try:
            async with sse_client(url=self.url, httpx_client=http_client) as (
                read_stream,
                write_stream,
            ):
                session = ClientSession(read_stream, write_stream)
                async with session:
                    await session.initialize()
                    yield session
        finally:
            await http_client.aclose()

    def __repr__(self) -> str:
        return f"StreamableHttpMCPTransport(url={self.url!r})"


class SSEHttpMCPTransport(MCPTransport):
    """通过 HTTP SSE 协议与 MCP 服务器通信。

    这是 StreamableHttpMCPTransport 的别名，使用相同的 SSE 传输。
    """

    def __init__(
        self,
        url: str,
        auth: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self._transport = StreamableHttpMCPTransport(url, auth, headers)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession, None]:
        """创建 SSE-HTTP 会话。"""
        async with self._transport.session() as session:
            yield session

    def __repr__(self) -> str:
        return f"SSEHttpMCPTransport(url={self._transport.url!r})"
