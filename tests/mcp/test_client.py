"""MCP 客户端测试."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_agents.mcp.client import MCPClient, MCPTool
from agentic_agents.mcp.transport import StdioMCPTransport


class TestMCPTool:
    """MCPTool 测试。"""

    def test_from_mcp_tool(self):
        """测试从 mcp.types.Tool 创建 MCPTool。"""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }

        tool = MCPTool.from_mcp_tool(mock_tool)
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == mock_tool.inputSchema
        assert tool._raw is mock_tool


class TestMCPClient:
    """MCPClient 测试。"""

    def test_init(self):
        """测试初始化。"""
        transport = StdioMCPTransport(command="npx", args=["server"])
        client = MCPClient(name="test_server", transport=transport)

        assert client.name == "test_server"
        assert client.transport is transport
        assert client._session is None

    def test_session_not_connected_error(self):
        """测试未连接时访问 session 抛出错误。"""
        transport = StdioMCPTransport(command="npx", args=["server"])
        client = MCPClient(name="test_server", transport=transport)

        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.session

    def test_repr(self):
        """测试字符串表示。"""
        transport = StdioMCPTransport(command="npx", args=["server"])
        client = MCPClient(name="test_server", transport=transport)
        assert "test_server" in repr(client)
        assert "StdioMCPTransport" in repr(client)

    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        """测试 async context manager。"""
        transport = StdioMCPTransport(command="npx", args=["server"])
        client = MCPClient(name="test_server", transport=transport)

        # 注意：由于没有真实的 MCP 服务器，这个测试只验证上下文管理器的行为
        # 实际连接会失败，但这不影响对上下文管理器本身的测试
        with pytest.raises(Exception):  # 连接会失败
            async with client:
                pass
