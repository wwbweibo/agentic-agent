"""Agent 单元测试 - 覆盖 Agent 核心逻辑."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_agents.agents.base import Agent
from agentic_agents.agents.handoff import create_transfer_tool
from agentic_agents.llm.base import AgentMessage, ChatResult, ToolCall
from agentic_agents.tools.base import AgentTool, create_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str = "test_tool", func=None) -> AgentTool:
    return create_tool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        func=func or (lambda x: f"result:{x}"),
    )


def _make_llm_mock(responses: list[ChatResult]) -> AsyncMock:
    """创建一个按顺序返回 ChatResult 的 mock LLM."""
    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=responses)
    llm.supports_tools.return_value = True
    return llm


def _text_result(content: str) -> ChatResult:
    """创建一个纯文本 ChatResult（无 tool_calls）."""
    return ChatResult(
        message=AgentMessage(role="assistant", content=content, tool_calls=[]),
        raw=None,
    )


def _tool_call_result(
    content: str,
    tool_calls: list[ToolCall],
) -> ChatResult:
    """创建一个带 tool_calls 的 ChatResult."""
    return ChatResult(
        message=AgentMessage(role="assistant", content=content, tool_calls=tool_calls),
        raw=None,
    )


# ---------------------------------------------------------------------------
# _tool_definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_empty_tools(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        assert agent._tool_definitions() == []

    def test_returns_tool_dicts(self):
        tools = [_make_tool("t1"), _make_tool("t2")]
        agent = Agent("a", "d", AsyncMock(), tools, "prompt")
        defs = agent._tool_definitions()
        assert len(defs) == 2
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "t1"


# ---------------------------------------------------------------------------
# _execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_tool_not_found(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        tc = ToolCall(id="1", name="missing", arguments={"x": "v"})
        result = agent._execute_tool(tc)
        assert "not found" in result

    def test_sync_tool_execution(self):
        tool = _make_tool("echo", func=lambda x: f"echo:{x}")
        agent = Agent("a", "d", AsyncMock(), [tool], "prompt")
        tc = ToolCall(id="1", name="echo", arguments={"x": "hello"})
        result = agent._execute_tool(tc)
        assert result == "echo:hello"

    def test_tool_exception_returns_error(self):
        def failing(x):
            raise ValueError("boom")

        tool = _make_tool("fail", func=failing)
        agent = Agent("a", "d", AsyncMock(), [tool], "prompt")
        tc = ToolCall(id="1", name="fail", arguments={"x": "v"})
        result = agent._execute_tool(tc)
        assert "Error executing tool" in result
        assert "boom" in result


# ---------------------------------------------------------------------------
# _is_transfer_call / _extract_transfer_target
# ---------------------------------------------------------------------------

class TestTransferDetection:
    def test_is_transfer_call_true(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        tc = ToolCall(id="1", name="transfer_to_Router", arguments={})
        assert agent._is_transfer_call(tc) is True

    def test_is_transfer_call_false(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        tc = ToolCall(id="1", name="web_search", arguments={})
        assert agent._is_transfer_call(tc) is False

    def test_extract_transfer_target(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        tc = ToolCall(id="1", name="transfer_to_Agent1", arguments={"reason": "done"})
        target, reason = agent._extract_transfer_target(tc)
        assert target == "Agent1"
        assert reason == "done"

    def test_extract_transfer_target_no_reason(self):
        agent = Agent("a", "d", AsyncMock(), [], "prompt")
        tc = ToolCall(id="1", name="transfer_to_Agent1", arguments={})
        target, reason = agent._extract_transfer_target(tc)
        assert target == "Agent1"
        assert reason == "No reason provided"


# ---------------------------------------------------------------------------
# astream
# ---------------------------------------------------------------------------

class TestAstream:
    @pytest.mark.asyncio
    async def test_text_only_response(self):
        """LLM 返回纯文本，Agent 产出 text 事件后结束."""
        llm = _make_llm_mock([_text_result("Hello!")])
        agent = Agent("TestAgent", "d", llm, [], "system prompt")

        events = [e async for e in agent.astream({"messages": []})]
        assert len(events) == 1
        assert events[0]["type"] == "text"
        assert events[0]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_and_text(self):
        """LLM 先返回 tool_call，执行后再返回文本."""
        tool = _make_tool("echo", func=lambda x: f"echo:{x}")
        llm = _make_llm_mock([
            _tool_call_result("Let me search", [
                ToolCall(id="tc1", name="echo", arguments={"x": "hi"}),
            ]),
            _text_result("Done"),
        ])
        agent = Agent("TestAgent", "d", llm, [tool], "prompt")

        events = [e async for e in agent.astream({"messages": []})]
        types = [e["type"] for e in events]
        assert "text" in types
        assert "tool_result" in types
        # tool_result 的内容是 echo:hi
        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["content"] == "echo:hi"
        assert tool_event["tool_name"] == "echo"

    @pytest.mark.asyncio
    async def test_transfer_event(self):
        """LLM 返回 transfer 调用，Agent 产出 transfer 事件."""
        transfer = create_transfer_tool("Router", "transfer back")
        llm = _make_llm_mock([
            _tool_call_result("Transferring", [
                ToolCall(id="tc1", name="transfer_to_Router", arguments={"reason": "task done"}),
            ]),
        ])
        agent = Agent("Worker", "d", llm, [transfer], "prompt")

        events = [e async for e in agent.astream({"messages": []})]
        transfer_events = [e for e in events if e["type"] == "transfer"]
        assert len(transfer_events) == 1
        assert transfer_events[0]["to_agent"] == "Router"
        assert transfer_events[0]["reason"] == "task done"

    @pytest.mark.asyncio
    async def test_max_epochs_forces_transfer(self):
        """超过 max_epochs 后强制转回 Router."""
        # LLM 每次都返回 tool_call 使循环持续
        tool = _make_tool("loop", func=lambda x: "looping")

        def make_tool_call_response():
            return _tool_call_result("thinking", [
                ToolCall(id="tc1", name="loop", arguments={"x": "v"}),
            ])

        # max_epochs=2，需要 3 个响应（前 2 次循环后第 3 次触发 max_epochs）
        llm = _make_llm_mock([make_tool_call_response() for _ in range(3)])
        agent = Agent("Worker", "d", llm, [tool], "prompt", max_epochs=2)

        events = [e async for e in agent.astream({"messages": []})]
        transfer_events = [e for e in events if e["type"] == "transfer"]
        assert len(transfer_events) == 1
        assert transfer_events[0]["to_agent"] == "Router"
        assert "Maximum epochs" in transfer_events[0]["reason"]

    @pytest.mark.asyncio
    async def test_tool_call_arguments_serialized_as_string(self):
        """tool_calls 中的 arguments 应序列化为 JSON 字符串."""
        tool = _make_tool("echo", func=lambda x: f"echo:{x}")
        llm = _make_llm_mock([
            _tool_call_result("", [
                ToolCall(id="tc1", name="echo", arguments={"x": "val"}),
            ]),
            _text_result("done"),
        ])
        agent = Agent("A", "d", llm, [tool], "prompt")

        [e async for e in agent.astream({"messages": []})]

        # 检查第二次 LLM 调用时消息中 tool_calls 的 arguments 是字符串
        second_call_messages = llm.chat.call_args_list[1][1]["messages"]
        assistant_msg = next(m for m in second_call_messages if m.get("tool_calls"))
        args = assistant_msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)

    @pytest.mark.asyncio
    async def test_tool_call_includes_type_function(self):
        """tool_calls 中应包含 type: function 字段."""
        tool = _make_tool("echo", func=lambda x: f"echo:{x}")
        llm = _make_llm_mock([
            _tool_call_result("", [
                ToolCall(id="tc1", name="echo", arguments={"x": "val"}),
            ]),
            _text_result("done"),
        ])
        agent = Agent("A", "d", llm, [tool], "prompt")

        [e async for e in agent.astream({"messages": []})]

        second_call_messages = llm.chat.call_args_list[1][1]["messages"]
        assistant_msg = next(m for m in second_call_messages if m.get("tool_calls"))
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"

    @pytest.mark.asyncio
    async def test_invoke_collects_events(self):
        """invoke 方法收集所有事件."""
        llm = _make_llm_mock([_text_result("hi")])
        agent = Agent("A", "d", llm, [], "prompt")
        result = await agent.invoke({"messages": []})
        assert "events" in result
        assert len(result["events"]) == 1


# ---------------------------------------------------------------------------
# create_transfer_tool
# ---------------------------------------------------------------------------

class TestCreateTransferTool:
    def test_creates_tool_with_correct_name(self):
        t = create_transfer_tool("Agent1", "Transfer to Agent1")
        assert t.name == "transfer_to_Agent1"
        assert isinstance(t, AgentTool)

    def test_transfer_func_returns_marker(self):
        t = create_transfer_tool("Agent1", "Transfer to Agent1")
        result = t.func(reason="done with task")
        assert result == "TRANSFER_TO:Agent1:done with task"

    def test_transfer_func_empty_reason(self):
        t = create_transfer_tool("Router", "back to router")
        result = t.func()
        assert result == "TRANSFER_TO:Router:"

    def test_to_dict_format(self):
        t = create_transfer_tool("Agent1", "desc")
        d = t.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "transfer_to_Agent1"
        assert "reason" in d["function"]["parameters"]["properties"]
