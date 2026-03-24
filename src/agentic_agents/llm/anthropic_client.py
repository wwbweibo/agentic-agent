import json
from typing import Any

from anthropic import AsyncAnthropic

from .base import AgentMessage, ChatResult, LLMClient, ToolCall


class AnthropicClient(LLMClient):
    """使用 Anthropic SDK 的 LLM 客户端 (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, **kwargs)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        **kwargs,
    ) -> ChatResult:
        # 将 OpenAI 格式消息转为 Anthropic 格式
        anthropic_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system = msg.get("content", "")
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })
            elif role == "assistant":
                content: list[dict] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        content.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": tc["function"]["arguments"]
                                if isinstance(tc["function"]["arguments"], dict)
                                else json.loads(tc["function"]["arguments"]),
                        })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content or [{"type": "text", "text": ""}],
                })
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })

        params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [self._convert_tool(t) for t in tools]

        response = await self._client.messages.create(**params)

        # 解析响应
        tool_calls: list[ToolCall] = []
        content_text = ""

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        agent_msg = AgentMessage(
            role="assistant",
            content=content_text,
            tool_calls=tool_calls,
        )
        return ChatResult(message=agent_msg, raw=response)

    def supports_tools(self) -> bool:
        return True

    def _convert_tool(self, tool: dict) -> dict:
        """将 OpenAI 格式工具定义转为 Anthropic 格式."""
        func = tool.get("function", tool)
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        }

    async def close(self) -> None:
        await self._client.close()
