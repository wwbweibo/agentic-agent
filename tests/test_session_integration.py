"""AgentSession 集成测试（不依赖外部 LLM）."""

import pytest

from agentic_agents.session import (
    AgentSession,
    LocalFileSessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


# ---------------------------------------------------------------------------
# Mock Agent：模拟真实 Agent.astream() 行为
# ---------------------------------------------------------------------------

class MockAgent:
    """不调用 LLM，只按预设行为 yield 事件的 Mock Agent."""

    def __init__(self, name: str, events: list[dict]):
        self.name = name
        self._events = events

    async def astream(self, state: dict):
        for event in self._events:
            yield event


class LoopingAgent:
    """循环后返回（不转移）的 Mock Agent，用于触发 session max_epochs.

    每次 astream() 调用产生多次 yield 后返回。session 级别的 max_epochs 在
    astream() 返回后检查（epoch++ 后再检查），因此需要 agent 返回 max_epochs+1 次
    才能触发 session 错误。
    """

    def __init__(self, name: str, epochs_per_call: int = 2):
        self.name = name
        self._epochs_per_call = epochs_per_call
        self.max_epochs = 50

    async def astream(self, state: dict):
        for i in range(self._epochs_per_call):
            yield {"type": "text", "agent": self.name, "content": f"thinking {i}"}
        # 不转移，正常返回 → session 检查 max_epochs


async def noop_factory(agent_config, session_id: str, skill_dir: str, **kwargs) -> dict[str, MockAgent]:
    return {
        "Router": MockAgent("Router", []),
        "Agent1": MockAgent("Agent1", []),
        "Agent2": MockAgent("Agent2", []),
    }


# ---------------------------------------------------------------------------
# 基本流程测试
# ---------------------------------------------------------------------------

class TestAgentSessionBasic:
    """测试 AgentSession 基本生命周期."""

    @pytest.mark.asyncio
    async def test_process_empty_input_yields_nothing(self, mock_agent_factory):
        session = AgentSession(
            
            session_id="s1",
            agent_factory=mock_agent_factory,
        )
        results = [e async for e in session.process_message("")]
        assert results == []

    @pytest.mark.asyncio
    async def test_agent_not_found_yields_error(self, mock_agent_factory):
        session = AgentSession(
            
            session_id="s1",
            agent_factory=mock_agent_factory,
            entry_agent="NonExistent",
            max_epochs=1,
        )
        results = [e async for e in session.process_message("hello")]
        assert any(e["resp_type"] == "error" for e in results)

    @pytest.mark.asyncio
    async def test_max_epochs_yields_error(self):
        """LoopingAgent 返回后，epoch 已达上限，触发 max_epochs 错误."""
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {"Router": LoopingAgent("Router", epochs_per_call=1)}

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            max_epochs=1,
        )
        results = [e async for e in session.process_message("hello")]
        assert any(e["resp_type"] == "error" and "epochs reached" in e["content"] for e in results)

    @pytest.mark.asyncio
    async def test_success_callback(self):
        """MockAgent 不 yield 任何事件，session 正常结束并触发 success_callback."""
        called = {}

        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {"Router": MockAgent("Router", [])}

        async def on_success(session, message, messages, response):
            called["fired"] = True

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            success_callback=on_success,
        )
        [e async for e in session.process_message("hello")]
        assert called.get("fired") is True

    @pytest.mark.asyncio
    async def test_error_callback_on_agent_not_found(self):
        """entry_agent 不存在时触发 error_callback."""
        called = {}

        async def on_error(session, message, messages, response):
            called["fired"] = True
            called["message"] = message

        session = AgentSession(
            
            session_id="s1",
            agent_factory=noop_factory,
            entry_agent="NonExistent",
            error_callback=on_error,
            max_epochs=1,
        )
        [e async for e in session.process_message("hello")]
        assert called.get("fired") is True
        assert "not found" in called["message"]

    @pytest.mark.asyncio
    async def test_text_event_forwarded(self):
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "text", "agent": "Router", "content": "Hello!"},
                ]),
            }

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            max_epochs=2,
        )
        results = [e async for e in session.process_message("hi")]
        assert any(
            e.get("resp_type") == "text"
            and e.get("agent") == "Router"
            and e.get("content") == "Hello!"
            for e in results
        )

    @pytest.mark.asyncio
    async def test_tool_result_event_forwarded(self):
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "tool_result", "agent": "Router", "content": "tool output"},
                ]),
            }

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            max_epochs=2,
        )
        results = [e async for e in session.process_message("run tool")]
        assert any(
            e.get("resp_type") == "tool_result"
            and e.get("agent") == "Router"
            and e.get("content") == "tool output"
            for e in results
        )


# ---------------------------------------------------------------------------
# Storage 集成测试
# ---------------------------------------------------------------------------

class TestAgentSessionStorageIntegration:
    """测试 AgentSession 与不同存储后端的集成."""

    @pytest.mark.asyncio
    async def test_memory_storage_persists_messages_across_turns(self):
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "text", "agent": "Router", "content": "response1"},
                ]),
            }

        storage = SessionStorage("s1")

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            storage=storage,
            max_epochs=1,
        )

        # 第一轮
        [e async for e in session.process_message("turn1")]
        saved_after_turn1 = list(storage._messages)

        # 第二轮：session 重用同一 storage，继续累积消息
        session2 = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            storage=storage,
            max_epochs=1,
        )
        [e async for e in session2.process_message("turn2")]
        # 内存存储：两轮累积了 user + assistant 消息
        assert len(storage._messages) >= 2

    @pytest.mark.asyncio
    async def test_local_file_storage_round_trip(self, tmp_path):
        storage_dir = str(tmp_path / "sessions")

        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "text", "agent": "Router", "content": "saved response"},
                ]),
            }

        storage = LocalFileSessionStorage("s2", storage_dir=storage_dir)
        session = AgentSession(
            
            session_id="s2",
            agent_factory=factory,
            storage=storage,
            max_epochs=1,
        )
        [e async for e in session.process_message("hello")]

        # 重启 session，从文件加载消息
        storage2 = LocalFileSessionStorage("s2", storage_dir=storage_dir)
        session2 = AgentSession(
            
            session_id="s2",
            agent_factory=factory,
            storage=storage2,
            max_epochs=1,
        )
        [e async for e in session2.process_message("world")]

        loaded = await storage2.load_messages()
        assert len(loaded) >= 2  # user + assistant 消息

    @pytest.mark.asyncio
    async def test_sqlite_storage_round_trip(self, tmp_path):
        db_path = str(tmp_path / "sessions" / "test.db")

        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "text", "agent": "Router", "content": "sqlite response"},
                ]),
            }

        storage = SQLiteSessionStorage("s3", db_path=db_path)
        session = AgentSession(
            
            session_id="s3",
            agent_factory=factory,
            storage=storage,
            max_epochs=1,
        )
        [e async for e in session.process_message("hello")]

        # 重启 session，从数据库加载消息
        storage2 = SQLiteSessionStorage("s3", db_path=db_path)
        session2 = AgentSession(
            
            session_id="s3",
            agent_factory=factory,
            storage=storage2,
            max_epochs=1,
        )
        [e async for e in session2.process_message("world")]

        loaded = await storage2.load_messages()
        assert len(loaded) >= 2


# ---------------------------------------------------------------------------
# Transfer 流程测试
# ---------------------------------------------------------------------------

class TestAgentSessionTransfer:
    """测试 Agent 之间的转移流程."""

    @pytest.mark.asyncio
    async def test_transfer_yields_transfer_event(self):
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {"type": "text", "agent": "Router", "content": "transferring..."},
                    {
                        "type": "transfer",
                        "from_agent": "Router",
                        "to_agent": "Agent1",
                        "reason": "task delegation",
                        "tool_result": "TRANSFER_TO:Agent1:task delegation",
                    },
                ]),
                "Agent1": MockAgent("Agent1", [
                    {"type": "text", "agent": "Agent1", "content": "done"},
                ]),
            }

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            max_epochs=5,
        )
        results = [e async for e in session.process_message("delegate task")]

        assert any(e["resp_type"] == "transfer" and e["to_agent"] == "Agent1" for e in results)
        assert any(e["resp_type"] == "finished" for e in results)
        assert session.active_agent_name == "Agent1"

    @pytest.mark.asyncio
    async def test_transfer_to_unknown_agent_no_finished(self):
        """转移到不存在的 Agent，目标不在 agents 字典中，yield error."""
        async def factory(agent_config, session_id: str, skill_dir: str, **kwargs):
            return {
                "Router": MockAgent("Router", [
                    {
                        "type": "transfer",
                        "from_agent": "Router",
                        "to_agent": "NonExistent",
                        "reason": "oops",
                        "tool_result": "TRANSFER_TO:NonExistent:oops",
                    },
                ]),
            }

        session = AgentSession(
            
            session_id="s1",
            agent_factory=factory,
            max_epochs=5,
        )
        results = [e async for e in session.process_message("go nowhere")]
        # 转移目标不存在，yield error 而非 finished
        assert any(e["resp_type"] == "error" for e in results)


# ---------------------------------------------------------------------------
# _compress_context_if_needed 测试
# ---------------------------------------------------------------------------

class TestCompressContext:
    """测试历史记录压缩逻辑."""

    @pytest.mark.asyncio
    async def test_compress_skips_when_not_router(self):
        """转移目标不是 Router 时不压缩."""
        storage = SessionStorage("s1")
        session = AgentSession(
            
            session_id="s1",
            agent_factory=noop_factory,
            storage=storage,
        )
        session.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        original_len = len(session.messages)

        await session._compress_context_if_needed("Agent1", "Agent2", "reason")
        # 目标不是 Router，不应压缩
        assert len(session.messages) == original_len

    @pytest.mark.asyncio
    async def test_compress_adds_summary_message(self):
        """转移回 Router 时，截断到转交点，添加摘要，再追加转交消息."""
        storage = SessionStorage("s1")
        session = AgentSession(
            
            session_id="s1",
            agent_factory=noop_factory,
            storage=storage,
        )
        session.messages = [
            {"role": "user", "content": "hello"},  # 0
            {"role": "assistant", "content": "r1"},  # 1
            {"role": "tool", "content": "TRANSFER_TO:Agent1:reason"},  # 2 ← 转交点
            {"role": "assistant", "content": "agent1 result"},  # 3
            {"role": "tool", "content": "TRANSFER_TO:Router:done"},  # 4
        ]

        await session._compress_context_if_needed("Agent1", "Router", "done")

        # 转交点消息保留（idx 2）
        assert len(session.messages) >= 3
        # 最后一条来自 transfer_back_msg
        last = session.messages[-1]
        assert last["role"] in ("assistant", "user", "tool")
