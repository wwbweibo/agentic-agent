"""
AgentTool - 替代 LangChain BaseTool 的工具定义。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

# 支持 sync 和 async 函数
ToolFunc = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass
class AgentTool:
    """标准化的工具定义."""
    name: str
    description: str
    func: ToolFunc
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为 OpenAI/Anthropic 格式的工具定义."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> Any:
        """执行工具函数."""
        import asyncio
        result = self.func(**kwargs)
        if asyncio.iscoroutine(result):
            return result
        return result


def create_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    func: ToolFunc,
    tags: list[str] | None = None,
) -> AgentTool:
    """创建一个工具定义.

    Args:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 格式的参数定义
        func: 工具函数（支持 sync 或 async）
        tags: 标签列表
    """
    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        func=func,
        tags=tags or [],
    )
