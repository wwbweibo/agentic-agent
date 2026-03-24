"""SessionStorage 单元测试."""

import json
from pathlib import Path

import pytest

from agentic_agents.session import (
    LocalFileSessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


# =============================================================================
# SessionStorage 基类
# =============================================================================

class TestSessionStorage:
    """测试基类内存存储的行为."""

    @pytest.fixture
    def storage(self):
        return SessionStorage("test-sid", ttl=3600)

    @pytest.mark.asyncio
    async def test_save_and_load_messages(self, storage):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        await storage.save_messages(messages)
        loaded = await storage.load_messages()
        assert loaded == messages

    @pytest.mark.asyncio
    async def test_save_and_load_response(self, storage):
        response = [{"resp_type": "text", "content": "hello"}]
        await storage.save_response(response)
        loaded = await storage.load_response()
        assert loaded == response

    @pytest.mark.asyncio
    async def test_load_empty_returns_empty_list(self, storage):
        loaded = await storage.load_messages()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_empty_response_returns_none(self, storage):
        loaded = await storage.load_response()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_clear(self, storage):
        await storage.save_messages([{"role": "user", "content": "x"}])
        await storage.save_response([{"resp_type": "text", "content": "x"}])
        await storage.clear()
        assert await storage.load_messages() == []
        assert await storage.load_response() is None

    @pytest.mark.asyncio
    async def test_save_empty_messages_does_nothing(self, storage):
        await storage.save_messages([])
        assert await storage.load_messages() == []

    @pytest.mark.asyncio
    async def test_session_id_and_ttl(self):
        storage = SessionStorage("my-sid", ttl=86400)
        assert storage.session_id == "my-sid"
        assert storage.ttl == 86400


# =============================================================================
# LocalFileSessionStorage
# =============================================================================

class TestLocalFileSessionStorage:
    """测试本地文件存储."""

    @pytest.mark.asyncio
    async def test_save_and_load_messages(self, local_file_storage):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        await local_file_storage.save_messages(messages)
        loaded = await local_file_storage.load_messages()
        assert loaded == messages

    @pytest.mark.asyncio
    async def test_save_and_load_response(self, local_file_storage):
        response = [{"resp_type": "text", "content": "result"}]
        await local_file_storage.save_response(response)
        loaded = await local_file_storage.load_response()
        assert loaded == response

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_empty(self, local_file_storage):
        loaded = await local_file_storage.load_messages()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_response_nonexistent_returns_none(self, local_file_storage):
        loaded = await local_file_storage.load_response()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_save_messages_creates_meta_file(self, local_file_storage):
        await local_file_storage.save_messages([{"role": "user", "content": "x"}])
        meta = json.loads(local_file_storage._meta_path.read_text(encoding="utf-8"))
        assert "updated_at" in meta
        assert "expire_at" in meta
        assert meta["expire_at"] > meta["updated_at"]

    @pytest.mark.asyncio
    async def test_save_empty_messages_skips_write(self, local_file_storage):
        await local_file_storage.save_messages([])
        assert not local_file_storage._messages_path.exists()

    @pytest.mark.asyncio
    async def test_clear_removes_all_files(self, local_file_storage):
        await local_file_storage.save_messages([{"role": "user", "content": "x"}])
        await local_file_storage.save_response([{"resp_type": "text", "content": "x"}])
        await local_file_storage.clear()
        assert not local_file_storage._messages_path.exists()
        assert not local_file_storage._response_path.exists()
        assert not local_file_storage._meta_path.exists()

    @pytest.mark.asyncio
    async def test_overwrite_messages(self, local_file_storage):
        await local_file_storage.save_messages([{"role": "user", "content": "v1"}])
        await local_file_storage.save_messages([{"role": "user", "content": "v2"}])
        loaded = await local_file_storage.load_messages()
        assert loaded == [{"role": "user", "content": "v2"}]

    @pytest.mark.asyncio
    async def test_storage_dir_created(self, tmp_path):
        storage = LocalFileSessionStorage("new-session", storage_dir=str(tmp_path / "sub"))
        assert storage._storage_dir.exists()

    @pytest.mark.asyncio
    async def test_messages_file_is_valid_json(self, local_file_storage):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "你好"},
        ]
        await local_file_storage.save_messages(messages)
        loaded = json.loads(local_file_storage._messages_path.read_text(encoding="utf-8"))
        assert loaded == messages


# =============================================================================
# SQLiteSessionStorage
# =============================================================================

class TestSQLiteSessionStorage:
    """测试 SQLite 存储."""

    @pytest.mark.asyncio
    async def test_save_and_load_messages(self, sqlite_storage):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        await sqlite_storage.save_messages(messages)
        loaded = await sqlite_storage.load_messages()
        assert loaded == messages

    @pytest.mark.asyncio
    async def test_save_and_load_response(self, sqlite_storage):
        response = [{"resp_type": "text", "content": "result"}]
        await sqlite_storage.save_response(response)
        loaded = await sqlite_storage.load_response()
        assert loaded == response

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_empty(self, sqlite_storage):
        loaded = await sqlite_storage.load_messages()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_load_response_nonexistent_returns_none(self, sqlite_storage):
        loaded = await sqlite_storage.load_response()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_save_empty_messages_does_nothing(self, sqlite_storage):
        await sqlite_storage.save_messages([])
        loaded = await sqlite_storage.load_messages()
        assert loaded == []

    @pytest.mark.asyncio
    async def test_clear(self, sqlite_storage):
        await sqlite_storage.save_messages([{"role": "user", "content": "x"}])
        await sqlite_storage.save_response([{"resp_type": "text", "content": "x"}])
        await sqlite_storage.clear()
        assert await sqlite_storage.load_messages() == []
        assert await sqlite_storage.load_response() is None

    @pytest.mark.asyncio
    async def test_overwrite_messages(self, sqlite_storage):
        await sqlite_storage.save_messages([{"role": "user", "content": "v1"}])
        await sqlite_storage.save_messages([{"role": "user", "content": "v2"}])
        loaded = await sqlite_storage.load_messages()
        assert loaded == [{"role": "user", "content": "v2"}]

    @pytest.mark.asyncio
    async def test_db_dir_created(self, tmp_path):
        db = tmp_path / "sub" / "test.db"
        storage = SQLiteSessionStorage("new-session", db_path=str(db))
        assert storage._db_path.parent.exists()

    @pytest.mark.asyncio
    async def test_concurrent_save_same_session(self, sqlite_storage):
        """多个并发写操作不应产生重复记录（每次全量覆盖）."""
        await sqlite_storage.save_messages([
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ])
        await sqlite_storage.save_messages([{"role": "user", "content": "msg3"}])
        loaded = await sqlite_storage.load_messages()
        assert loaded == [{"role": "user", "content": "msg3"}]

    @pytest.mark.asyncio
    async def test_response_replace_behavior(self, sqlite_storage):
        await sqlite_storage.save_response([{"resp_type": "a", "content": "v1"}])
        await sqlite_storage.save_response([{"resp_type": "b", "content": "v2"}])
        loaded = await sqlite_storage.load_response()
        assert loaded == [{"resp_type": "b", "content": "v2"}]
