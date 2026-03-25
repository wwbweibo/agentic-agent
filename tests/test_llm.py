"""LLM 客户端单元测试 - OpenAIClient 和 AnthropicClient (mock)."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_agents.llm.base import AgentMessage, ChatResult, LLMClient, ToolCall
from agentic_agents.llm.openai_client import OpenAIClient
from agentic_agents.llm.anthropic_client import AnthropicClient


# ---------------------------------------------------------------------------
# ToolCall / AgentMessage / ChatResult 数据类
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="tc1", name="search", arguments={"q": "hello"})
        d = tc.to_dict()
        assert d == {"id": "tc1", "name": "search", "arguments": {"q": "hello"}}


class TestAgentMessage:
    def test_to_dict_simple(self):
        msg = AgentMessage(role="assistant", content="hi")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "hi"
        assert "tool_calls" not in d

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="search", arguments={"q": "x"})
        msg = AgentMessage(role="assistant", content="let me search", tool_calls=[tc])
        d = msg.to_dict()
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "tc1"
        assert d["tool_calls"][0]["function"]["name"] == "search"

    def test_to_dict_with_tool_call_id(self):
        msg = AgentMessage(role="tool", content="result", tool_call_id="tc1")
        d = msg.to_dict()
        assert d["tool_call_id"] == "tc1"


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

class TestOpenAIClient:
    def test_init_default(self):
        with patch("agentic_agents.llm.openai_client.AsyncOpenAI"):
            client = OpenAIClient()
            assert client.model == "gpt-4o"

    def test_init_custom(self):
        with patch("agentic_agents.llm.openai_client.AsyncOpenAI"):
            client = OpenAIClient(model="gpt-4", api_key="key", base_url="http://localhost")
            assert client.model == "gpt-4"

    def test_supports_tools(self):
        with patch("agentic_agents.llm.openai_client.AsyncOpenAI"):
            client = OpenAIClient()
            assert client.supports_tools() is True

    @pytest.mark.asyncio
    async def test_chat_text_response(self):
        """测试纯文本响应的解析."""
        mock_msg = MagicMock()
        mock_msg.content = "Hello!"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agentic_agents.llm.openai_client.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            client = OpenAIClient(model="gpt-4o")
            result = await client.chat(messages=[{"role": "user", "content": "hi"}])

            assert isinstance(result, ChatResult)
            assert result.message.content == "Hello!"
            assert result.message.tool_calls == []

    @pytest.mark.asyncio
    async def test_chat_tool_call_response(self):
        """测试带 tool_calls 的响应解析."""
        mock_func = MagicMock()
        mock_func.name = "search"
        mock_func.arguments = json.dumps({"q": "test"})

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function = mock_func

        mock_msg = MagicMock()
        mock_msg.content = "Searching..."
        mock_msg.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agentic_agents.llm.openai_client.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            client = OpenAIClient()
            result = await client.chat(
                messages=[{"role": "user", "content": "search"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )

            assert len(result.message.tool_calls) == 1
            tc = result.message.tool_calls[0]
            assert tc.id == "call_123"
            assert tc.name == "search"
            assert tc.arguments == {"q": "test"}

    @pytest.mark.asyncio
    async def test_chat_passes_tools_to_api(self):
        """测试 tools 参数被正确传递."""
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agentic_agents.llm.openai_client.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            client = OpenAIClient()
            tools = [{"type": "function", "function": {"name": "t1"}}]
            await client.chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_chat_no_tools_omits_params(self):
        """没有 tools 时不传 tools 和 tool_choice."""
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_msg.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("agentic_agents.llm.openai_client.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            client = OpenAIClient()
            await client.chat(messages=[{"role": "user", "content": "hi"}])

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "tools" not in call_kwargs
            assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class TestAnthropicClient:
    def test_init_default(self):
        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic"):
            client = AnthropicClient()
            assert client.model == "claude-sonnet-4-20250514"

    def test_supports_tools(self):
        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic"):
            client = AnthropicClient()
            assert client.supports_tools() is True

    def test_convert_tool(self):
        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic"):
            client = AnthropicClient()
            openai_tool = {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
            anthropic_tool = client._convert_tool(openai_tool)
            assert anthropic_tool["name"] == "search"
            assert anthropic_tool["description"] == "Search the web"
            assert anthropic_tool["input_schema"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_chat_text_response(self):
        """测试纯文本响应解析."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello from Claude!"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            MockAnthropic.return_value = mock_client

            client = AnthropicClient()
            result = await client.chat(
                messages=[{"role": "user", "content": "hi"}],
            )

            assert result.message.content == "Hello from Claude!"
            assert result.message.tool_calls == []

    @pytest.mark.asyncio
    async def test_chat_tool_use_response(self):
        """测试 tool_use 响应解析."""
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"q": "test"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            MockAnthropic.return_value = mock_client

            client = AnthropicClient()
            result = await client.chat(
                messages=[{"role": "user", "content": "search"}],
            )

            assert len(result.message.tool_calls) == 1
            tc = result.message.tool_calls[0]
            assert tc.id == "toolu_123"
            assert tc.name == "search"
            assert tc.arguments == {"q": "test"}

    @pytest.mark.asyncio
    async def test_system_message_extracted(self):
        """测试 system 消息被提取到 system 参数."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "ok"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            MockAnthropic.return_value = mock_client

            client = AnthropicClient()
            await client.chat(messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "hi"},
            ])

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are an assistant."
            # system 消息不应出现在 messages 中
            for msg in call_kwargs["messages"]:
                assert msg["role"] != "system"

    @pytest.mark.asyncio
    async def test_tool_message_converted(self):
        """测试 tool 角色消息被转换为 Anthropic 的 tool_result 格式."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "ok"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("agentic_agents.llm.anthropic_client.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            MockAnthropic.return_value = mock_client

            client = AnthropicClient()
            await client.chat(messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc1", "function": {"name": "search", "arguments": {"q": "x"}}}
                ]},
                {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            ])

            call_kwargs = mock_client.messages.create.call_args[1]
            tool_msg = call_kwargs["messages"][2]
            assert tool_msg["role"] == "user"
            assert tool_msg["content"][0]["type"] == "tool_result"
            assert tool_msg["content"][0]["tool_use_id"] == "tc1"
