"""
AgentTool - 替代 LangChain BaseTool 的工具定义。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import inspect

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

# decorator 版本的工具定义
def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Callable[[ToolFunc], AgentTool]:
    """工具定义装饰器.

    Args:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 格式的参数定义
        tags: 标签列表
    """
    def decorator(func: ToolFunc) -> AgentTool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""
        tool_parameters = parameters
        if not tool_parameters:
            # 尝试从函数签名自动生成参数定义
            sig = inspect.signature(func)
            props = {}
            for param in sig.parameters.values():
                prop_name = param.name
                if param.annotation != inspect.Parameter.empty:
                    # 简单映射 Python 类型到 JSON Schema 类型
                    if param.annotation in (str, int, float, bool):
                        props[prop_name] = {"type": param.annotation.__name__}
                    else:
                        props[prop_name] = {"type": "string"}  # 默认复杂类型为字符串
                    # 检查默认值来确定是否必需
                    if param.default == inspect.Parameter.empty:
                        props[prop_name]["required"] = True
                else:
                    props[prop_name] = {"type": "string"}  # 无注解默认字符串
            tool_parameters = {
                "type": "object",
                "properties": props,
                "required": [k for k, v in props.items() if v.get("required")],
            }
        return create_tool(
            name=tool_name,
            description=tool_description,
            parameters=tool_parameters,
            func=func,
            tags=tags,
        )
    return decorator
