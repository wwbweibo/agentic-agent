"""
会话管理 - 管理多 Agent 系统的会话状态。
"""

import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from .agents.base import Agent
from .agents.helpers import extract_transfer_target, is_transfer_call


class SessionStorage:
    """会话状态存储基类，默认使用内存存储。

    子类可实现 Redis、数据库等持久化存储。
    """

    def __init__(self, session_id: str, ttl: int = 3600 * 24 * 7):
        self.session_id = session_id
        self.ttl = ttl
        self._messages: list[dict] = []
        self._response: list[dict[str, Any]] = []

    async def save_messages(self, messages: list[dict]) -> None:
        """保存消息列表到存储."""
        self._messages = messages

    async def save_response(self, response: list[dict[str, Any]]) -> None:
        """保存当前 Agent 的响应到存储."""
        self._response = response

    async def load_messages(self) -> list[dict]:
        """从存储加载消息列表."""
        return self._messages

    async def load_response(self) -> list[dict[str, Any]] | None:
        """从存储加载当前 Agent 的响应."""
        return self._response if self._response else None

    async def clear(self) -> None:
        """清除会话状态."""
        self._messages = []
        self._response = []


class RedisSessionStorage(SessionStorage):
    """使用 Redis 来保存会话状态."""

    def __init__(self, session_id: str, redis_client, ttl: int = 3600 * 24 * 7):
        super().__init__(session_id, ttl)
        self.key = f"agent_session:{session_id}"
        self.response_key = f"agent_response:{session_id}"
        self.redis = redis_client

    async def save_messages(self, messages: list[dict]) -> None:
        if not messages:
            return
        await self.redis.setex(self.key, self.ttl, json.dumps(messages, ensure_ascii=False))

    async def save_response(self, response: list[dict[str, Any]]) -> None:
        await self.redis.setex(self.response_key, self.ttl, json.dumps(response, ensure_ascii=False))

    async def load_messages(self) -> list[dict]:
        data_str = await self.redis.get(self.key)
        if not data_str:
            return []
        return json.loads(data_str)

    async def load_response(self) -> list[dict[str, Any]] | None:
        data_str = await self.redis.get(self.response_key)
        if not data_str:
            return None
        return json.loads(data_str)

    async def clear(self) -> None:
        await self.redis.delete(self.key)
        await self.redis.delete(self.response_key)


class AgentSession:
    """Agent 会话管理类，协调多个 Agent 的执行。"""

    def __init__(
        self,
        tenant_id: str,
        session_id: str,
        agent_factory: Callable[..., Awaitable[dict[str, Agent]]],
        skill_dir: str = "./skills",
        storage: SessionStorage | None = None,
        entry_agent: str = "Router",
        max_epochs: int = 10,
        error_callback: Callable[..., Awaitable[None]] | None = None,
        success_callback: Callable[..., Awaitable[None]] | None = None,
    ):
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.storage = storage or SessionStorage(session_id)
        self.messages: list[dict] = []  # OpenAI 格式的消息列表
        self.active_agent_name = entry_agent
        self.agents: dict[str, Agent] | None = None
        self.max_epochs = max_epochs
        self.agent_factory = agent_factory
        self.skill_dir = skill_dir
        self.error_callback = error_callback
        self.success_callback = success_callback

    async def on_message_updated(self) -> None:
        """当消息更新时调用，保存当前状态到存储."""
        await self.storage.save_messages(self.messages)

    async def _compress_context_if_needed(
        self, current_agent: str, next_agent: str, reason: str
    ) -> None:
        """当控制权交回关键节点时，对历史记录进行压缩.

        当控制权从 Agent 转移回 Router 时，压缩该 Agent 执行期间的消息，
        仅保留转移消息和汇总结果。
        """
        if next_agent != "Router":
            return

        transfer_back_msg: list[dict] = []
        last_msg_idx = len(self.messages) - 1
        cnt = 0

        # 向前遍历，找到从 Router -> current_agent 的转移消息
        for i in range(len(self.messages) - 1, -1, -1):
            cnt += 1
            msg = self.messages[i]
            if cnt <= 2:
                transfer_back_msg.append(msg)
            # 检测 tool 结果消息中的转移指令
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and content.startswith(f"TRANSFER_TO:{current_agent}"):
                    last_msg_idx = i
                    break

        # 保留从开始到 last_msg_idx 的消息
        self.messages = self.messages[:last_msg_idx + 1]

        # 添加汇总消息
        task_summary = ""
        if len(transfer_back_msg) > 1:
            tool_result = transfer_back_msg[1]
            if tool_result.get("role") == "tool":
                task_summary = tool_result.get("content", "")
        task_summary = task_summary if task_summary else reason
        self.messages.append({"role": "assistant", "content": task_summary})

        # 添加转移消息
        if transfer_back_msg and not reason.startswith("Maximum epochs reached"):
            self.messages.extend(reversed(transfer_back_msg))
        else:
            self.messages.append({
                "role": "user",
                "content": f"You have been transferred control by {current_agent}. Reason: {reason}. Please continue to process the task.",
            })
        await self.on_message_updated()

    async def handle_error(self, error_message: str) -> None:
        """处理错误，调用错误回调函数."""
        if self.error_callback:
            response = await self.storage.load_response()
            if not response:
                response = []
            response.append({"resp_type": "error", "content": error_message})
            await self.error_callback(self, error_message, self.messages, response)

    async def handle_success(self, success_message: str) -> None:
        """处理成功，调用成功回调函数."""
        if self.success_callback:
            response = await self.storage.load_response()
            if not response:
                response = []
            response.append({"resp_type": "success", "content": success_message})
            await self.success_callback(self, success_message, self.messages, response)

    async def process_message(self, user_input: str) -> AsyncGenerator[dict[str, Any], None]:
        """处理用户输入，yield 事件流。"""
        if not user_input.strip():
            return

        if not self.messages:
            self.messages = await self.storage.load_messages()

        if self.agents is None:
            self.agents = await self.agent_factory(
                self.tenant_id, self.session_id, self.skill_dir
            )

        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        await self.on_message_updated()

        epoch = 0
        while True:
            if epoch >= self.max_epochs:
                yield {"resp_type": "error", "content": "Maximum number of epochs reached."}
                break

            active_agent = self.agents.get(self.active_agent_name)
            if not active_agent:
                error_msg = f"Agent {self.active_agent_name} not found."
                await self.handle_error(error_msg)
                yield {"resp_type": "error", "content": error_msg}
                break

            yield {"resp_type": "status", "content": f"Agent {self.active_agent_name} is thinking..."}

            try:
                epoch += 1
                next_agent_name: str | None = None
                reason = "No reason provided"

                async for event in active_agent.astream({"messages": self.messages}):
                    event_type = event.get("type")
                    content = event.get("content", "")

                    if event_type == "text":
                        yield {
                            "resp_type": "text",
                            "agent": self.active_agent_name,
                            "content": content,
                        }
                    elif event_type == "tool_result":
                        yield {
                            "resp_type": "tool_result",
                            "agent": self.active_agent_name,
                            "content": content,
                        }
                    elif event_type == "transfer":
                        tool_result = event.get("tool_result", "")
                        next_agent_name = event.get("to_agent", "")
                        reason = event.get("reason", "No reason provided")

                        yield {
                            "resp_type": "transfer",
                            "from_agent": event.get("from_agent"),
                            "to_agent": next_agent_name,
                            "reason": reason,
                        }

                if next_agent_name and next_agent_name in self.agents:
                    self.active_agent_name = next_agent_name
                    await self._compress_context_if_needed(active_agent.name, next_agent_name, reason)
                    self.messages.append({
                        "role": "user",
                        "content": f"You have been transferred control by {active_agent.name}. Reason: {reason}. Please continue to process the task.",
                    })
                    await self.on_message_updated()
                    continue
                else:
                    if self.success_callback:
                        await self.handle_success("Task completed successfully.")
                    yield {"resp_type": "finished", "content": "Task processing completed."}
                    await self.storage.save_messages(self.messages)
                    break

            except Exception as e:
                import traceback
                traceback.print_exc()
                await self.handle_error(str(e))
                yield {"resp_type": "error", "content": str(e)}
                break
