"""MCP 工具适配器。

将 MCP 工具转换为 AgentTool，使其可被 Agent 调用。
"""
from __future__ import annotations

from typing import Any

from agentic_agents.mcp.client import MCPClient, MCPTool
from agentic_agents.tools.base import AgentTool, ToolFunc


class MCPToolAdapter:
    """将 MCP 工具适配为 AgentTool。

    包装一个 MCPTool 和 MCPClient，将 MCP 工具调用转换为 AgentTool 接口。

    Example:
        client = MCPClient(name="filesystem", transport=transport)
        await client.connect()

        tools = await client.list_tools()
        for mcp_tool in tools:
            adapter = MCPToolAdapter(mcp_tool, client)
            agent_tool = adapter.to_agent_tool()
            # agent_tool 可以直接传递给 Agent 使用
    """

    def __init__(self, mcp_tool: MCPTool, client: MCPClient):
        self.mcp_tool = mcp_tool
        self.client = client

    async def _execute(self, **kwargs: Any) -> str:
        """执行 MCP 工具。"""
        return await self.client.call_tool(self.mcp_tool.name, kwargs)

    def to_agent_tool(self) -> AgentTool:
        """转换为 AgentTool 格式。

        返回的 AgentTool:
        - name: MCP 工具名称
        - description: MCP 工具描述
        - parameters: MCP 工具的输入 schema
        - tags: 包含 ["mcp", client_name] 用于标识来源
        """
        return AgentTool(
            name=self.mcp_tool.name,
            description=self.mcp_tool.description,
            func=self._execute,  # type: ignore
            parameters=self.mcp_tool.input_schema,
            tags=["mcp", self.client.name],
        )

    def __repr__(self) -> str:
        return f"MCPToolAdapter(tool={self.mcp_tool.name!r}, client={self.client.name!r})"


def mcp_tools_to_agent_tools(
    mcp_tools: list[MCPTool], client: MCPClient
) -> list[AgentTool]:
    """将 MCP 工具列表转换为 AgentTool 列表。

    Args:
        mcp_tools: MCP 工具列表
        client: MCP 客户端实例

    Returns:
        AgentTool 列表
    """
    return [MCPToolAdapter(t, client).to_agent_tool() for t in mcp_tools]
