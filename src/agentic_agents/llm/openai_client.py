from typing import Any

from openai import AsyncOpenAI

from .base import AgentMessage, ChatResult, LLMClient, ToolCall


class OpenAIClient(LLMClient):
    """使用 OpenAI SDK 的 LLM 客户端."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ChatResult:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**params)

        choice = response.choices[0]
        msg = choice.message

        # 解析 tool_calls
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.function
                import json
                arguments = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
                tool_calls.append(ToolCall(
                    id=tc.id or "",
                    name=func.name or "",
                    arguments=arguments or {},
                ))

        agent_msg = AgentMessage(
            role="assistant",
            content=msg.content or "",
            tool_calls=tool_calls,
        )
        return ChatResult(message=agent_msg, raw=response)

    def supports_tools(self) -> bool:
        return True

    async def close(self) -> None:
        await self._client.close()
