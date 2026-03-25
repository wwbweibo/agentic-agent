"""pytest 配置与共享 fixtures."""

import asyncio
import shutil
from pathlib import Path

import pytest

from agentic_agents.session import (
    AgentSession,
    LocalFileSessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


@pytest.fixture
def temp_storage_dir(tmp_path):
    """提供临时目录用于 LocalFileSessionStorage 测试."""
    d = tmp_path / "sessions"
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_db_path(tmp_path):
    """提供临时路径用于 SQLiteSessionStorage 测试."""
    db = tmp_path / "sessions" / "test.db"
    yield str(db)
    shutil.rmtree(db.parent, ignore_errors=True)


@pytest.fixture
def memory_storage():
    """提供内存存储实例."""
    return SessionStorage("test-session")


@pytest.fixture
def local_file_storage(temp_storage_dir):
    """提供 LocalFileSessionStorage 实例."""
    storage = LocalFileSessionStorage("test-session", storage_dir=temp_storage_dir)
    return storage


@pytest.fixture
def sqlite_storage(temp_db_path):
    """提供 SQLiteSessionStorage 实例."""
    storage = SQLiteSessionStorage("test-session", db_path=temp_db_path)
    return storage


@pytest.fixture
def mock_agent_factory():
    """返回模拟的 agent_factory，不创建真实 agent."""

    class MockAgent:
        def __init__(self, name: str):
            self.name = name

        async def astream(self, state):
            return
            yield  # make it an async generator

    async def factory(agent_config, session_id: str, skill_dir: str, llm=None):
        return {
            "Router": MockAgent("Router"),
            "Agent1": MockAgent("Agent1"),
        }

    return factory


@pytest.fixture
def session_with_mocks(mock_agent_factory):
    """返回使用 mock factory 的 AgentSession（使用内存存储）."""
    return AgentSession(
        session_id="test-session",
        agent_factory=mock_agent_factory,
        skill_dir="./skills",
        storage=SessionStorage("test-session"),
    )
