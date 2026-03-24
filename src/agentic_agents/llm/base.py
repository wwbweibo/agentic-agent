from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolCall:
    """标准化的工具调用结构，兼容 OpenAI 和 Anthropic 格式."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class AgentMessage:
    """标准化的消息结构，替代 langchain_core.messages."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # 仅 tool 角色需要

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "function": {"arguments": tc.arguments, "name": tc.name}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ChatResult:
    """LLM 响应结果."""
    message: AgentMessage
    raw: Any  # 保留原始响应，用于调试


class LLMClient(ABC):
    """LLM 客户端抽象接口."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ChatResult:
        """发送对话请求并返回结果.

        Args:
            messages: OpenAI 格式的消息列表
            tools: OpenAI 格式的工具定义列表
        """
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """当前模型是否支持工具调用."""
        ...
