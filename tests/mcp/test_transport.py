"""MCP 传输层测试."""
import pytest

from agentic_agents.mcp.transport import (
    StdioMCPTransport,
    StreamableHttpMCPTransport,
    SSEHttpMCPTransport,
)


class TestStdioMCPTransport:
    """StdioMCPTransport 测试。"""

    def test_init(self):
        """测试初始化。"""
        transport = StdioMCPTransport(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert transport.command == "npx"
        assert transport.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    def test_init_with_env(self):
        """测试带环境变量的初始化。"""
        transport = StdioMCPTransport(
            command="npx",
            args=["-y", "server"],
            env={"DEBUG": "1"},
            cwd="/tmp",
        )
        assert transport.env == {"DEBUG": "1"}
        assert transport.cwd == "/tmp"

    def test_repr(self):
        """测试字符串表示。"""
        transport = StdioMCPTransport(command="npx", args=["server"])
        assert "npx" in repr(transport)
        assert "server" in repr(transport)


class TestStreamableHttpMCPTransport:
    """StreamableHttpMCPTransport 测试。"""

    def test_init(self):
        """测试初始化。"""
        transport = StreamableHttpMCPTransport(
            url="https://mcp.example.com/sse",
        )
        assert transport.url == "https://mcp.example.com/sse"
        assert transport.auth is None
        assert transport.headers == {}

    def test_init_with_auth(self):
        """测试带认证的初始化。"""
        transport = StreamableHttpMCPTransport(
            url="https://mcp.example.com/sse",
            auth="Bearer token123",
        )
        assert transport.auth == "Bearer token123"

    def test_init_with_headers(self):
        """测试带自定义头的初始化。"""
        transport = StreamableHttpMCPTransport(
            url="https://mcp.example.com/sse",
            headers={"X-Custom": "value"},
        )
        assert transport.headers == {"X-Custom": "value"}

    def test_repr(self):
        """测试字符串表示。"""
        transport = StreamableHttpMCPTransport(url="https://example.com/sse")
        assert "https://example.com/sse" in repr(transport)


class TestSSEHttpMCPTransport:
    """SSEHttpMCPTransport 测试。"""

    def test_init(self):
        """测试初始化。"""
        transport = SSEHttpMCPTransport(
            url="https://mcp.example.com/sse",
            auth="Bearer token",
        )
        assert transport._transport.url == "https://mcp.example.com/sse"
        assert transport._transport.auth == "Bearer token"

    def test_repr(self):
        """测试字符串表示。"""
        transport = SSEHttpMCPTransport(url="https://example.com/sse")
        assert "https://example.com/sse" in repr(transport)
