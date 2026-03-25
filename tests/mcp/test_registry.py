"""MCP 注册表测试."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_agents.mcp.registry import (
    connect_mcp_server,
    disconnect_mcp_server,
    get_mcp_tool,
    list_mcp_tools,
    clear_mcp_registry,
)
from agentic_agents.mcp.transport import StdioMCPTransport


@pytest.fixture
def registry():
    """每个测试前后清理注册表。"""
    # 清理前置状态
    from agentic_agents.mcp import registry as reg_module
    reg_module._clients.clear()
    reg_module._tools.clear()
    yield
    # 测试后也清理
    reg_module._clients.clear()
    reg_module._tools.clear()


class TestMCPRegistry:
    """MCP 注册表测试。"""

    @pytest.mark.asyncio
    async def test_connect_stdio_server(self, registry):
        """测试连接 stdio 服务器。"""
        from agentic_agents.mcp import registry as reg_module
        with patch("agentic_agents.mcp.registry.StdioMCPTransport") as mock_transport:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])

            with patch("agentic_agents.mcp.registry.MCPClient", return_value=mock_client):
                tools = await connect_mcp_server(
                    name="test_server",
                    transport="stdio",
                    command="npx",
                    args=["-y", "server"],
                )

                assert tools == []
                assert "test_server" in reg_module._clients

    @pytest.mark.asyncio
    async def test_connect_http_server(self, registry):
        """测试连接 HTTP 服务器。"""
        from agentic_agents.mcp import registry as reg_module
        with patch("agentic_agents.mcp.registry.StreamableHttpMCPTransport") as mock_transport:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(return_value=[])

            with patch("agentic_agents.mcp.registry.MCPClient", return_value=mock_client):
                tools = await connect_mcp_server(
                    name="http_server",
                    transport="http",
                    url="https://example.com/sse",
                    auth="Bearer token",
                )

                assert tools == []
                assert "http_server" in reg_module._clients

    @pytest.mark.asyncio
    async def test_connect_duplicate_name(self, registry):
        """测试重复连接同名服务器。"""
        from agentic_agents.mcp import registry as reg_module
        mock_client = MagicMock()
        reg_module._clients["existing"] = mock_client

        with pytest.raises(ValueError, match="already connected"):
            await connect_mcp_server(
                name="existing",
                transport="stdio",
                command="npx",
            )

    @pytest.mark.asyncio
    async def test_connect_invalid_transport(self, registry):
        """测试无效的传输协议。"""
        with pytest.raises(ValueError, match="Invalid transport"):
            await connect_mcp_server(
                name="test",
                transport="invalid",
            )

    @pytest.mark.asyncio
    async def test_disconnect_server(self, registry):
        """测试断开服务器连接。"""
        from agentic_agents.mcp import registry as reg_module
        mock_client = AsyncMock()
        reg_module._clients["test_server"] = mock_client

        await disconnect_mcp_server("test_server")

        assert "test_server" not in reg_module._clients
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, registry):
        """测试断开不存在的服务器。"""
        with pytest.raises(ValueError, match="not connected"):
            await disconnect_mcp_server("nonexistent")


class TestMCPToolGetters:
    """MCP 工具 getter 测试。"""

    def test_get_mcp_tool(self, registry):
        """测试获取单个工具。"""
        from agentic_agents.mcp import registry as reg_module
        mock_tool = MagicMock()
        reg_module._tools["test_tool"] = mock_tool

        result = get_mcp_tool("test_tool")
        assert result is mock_tool

    def test_get_nonexistent_tool(self, registry):
        """测试获取不存在的工具。"""
        result = get_mcp_tool("nonexistent")
        assert result is None

    def test_list_mcp_tools(self, registry):
        """测试列出所有工具。"""
        from agentic_agents.mcp import registry as reg_module
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()
        reg_module._tools["tool1"] = mock_tool1
        reg_module._tools["tool2"] = mock_tool2

        tools = list_mcp_tools()
        assert len(tools) == 2
        assert mock_tool1 in tools
        assert mock_tool2 in tools

    def test_list_empty_tools(self, registry):
        """测试列出空工具列表。"""
        tools = list_mcp_tools()
        assert tools == []
