"""
LLM 抽象层 - 支持 OpenAI 和 Anthropic SDK。

提供统一的 LLMClient 接口，两种实现：
- OpenAIClient: 使用 OpenAI SDK
- AnthropicClient: 使用 Anthropic SDK (Claude)
"""

from .base import AgentMessage, ChatResult, LLMClient, ToolCall
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

__all__ = [
    "AgentMessage",
    "ChatResult",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "ToolCall",
]
