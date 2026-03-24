"""
工具注册表 - 管理所有可用的工具。
"""

from .base import AgentTool

# 全局工具注册表
_registry: dict[str, AgentTool] = {}


def register_tool(tool: AgentTool) -> None:
    """注册一个工具到全局注册表."""
    _registry[tool.name] = tool


def get_tool(name: str) -> AgentTool | None:
    """根据名称获取工具."""
    return _registry.get(name)


def list_tools() -> list[AgentTool]:
    """列出所有已注册的工具."""
    return list(_registry.values())


def clear_registry() -> None:
    """清空注册表（主要用于测试）."""
    _registry.clear()
