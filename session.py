import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from langchain.messages import ToolMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnableConfig

from .agents.base import Agent
from .agents.helpers import is_tool_calls, is_transfer_call


class SessionStorage:
    """会话状态存储基类，默认使用内存存储。

    子类可实现 Redis、数据库等持久化存储。
    """

    def __init__(self, session_id: str, ttl: int = 3600 * 24 * 7):
        self.session_id = session_id
        self.ttl = ttl
        self._messages: list[BaseMessage] = []
        self._response: list[dict[str, Any]] = []

    async def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息列表到存储."""
        self._messages = messages

    async def save_response(self, response: list[dict[str, Any]]) -> None:
        """保存当前Agent的响应到存储."""
        self._response = response

    async def load_messages(self) -> list[BaseMessage]:
        """从存储加载消息列表."""
        return self._messages

    async def load_response(self) -> list[dict[str, Any]] | None:
        """从存储加载当前Agent的响应."""
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

    async def save_messages(self, messages: list[BaseMessage]) -> None:
        """保存消息列表到Redis."""
        if not messages:
            return
        data = messages_to_dict(messages)
        await self.redis.setex(self.key, self.ttl, json.dumps(data, ensure_ascii=False))

    async def save_response(self, response: list[dict[str, Any]]) -> None:
        """保存当前Agent的响应到Redis."""
        await self.redis.setex(self.response_key, self.ttl, json.dumps(response, ensure_ascii=False))

    async def load_messages(self) -> list[BaseMessage]:
        """从Redis加载消息列表."""
        data_str = await self.redis.get(self.key)
        if not data_str:
            return []
        data = json.loads(data_str)
        return messages_from_dict(data)

    async def load_response(self) -> list[dict[str, Any]] | None:
        """从Redis加载当前Agent的响应."""
        data_str = await self.redis.get(self.response_key)
        if not data_str:
            return None
        return json.loads(data_str)

    async def clear(self) -> None:
        """清除会话状态."""
        await self.redis.delete(self.key)
        await self.redis.delete(self.response_key)


class AgentSession:
    def __init__(self, tenant_id: str,
                session_id: str,
                agent_factory: Callable[[str, str, str, RunnableConfig|None], Awaitable[dict[str, Agent]]],
                skill_dir: str = './skills',
                storage: SessionStorage | None = None,
                entry_agent: str = "Router",
                runnable_config: RunnableConfig | None = None,
                error_callback: Callable[["AgentSession", str, list[dict[str, Any]], list[dict[str, Any]]], Awaitable[None]] | None = None, # noqa
                success_callback: Callable[["AgentSession", str, list[dict[str, Any]], list[dict[str, Any]]], Awaitable[None]] | None = None): # noqa
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.storage = storage or SessionStorage(session_id)
        self.messages = []
        self.active_agent_name = entry_agent
        self.agents = None  # Will be set when processing starts
        self.max_epochs = 10
        self.agent_factory = agent_factory
        self.skill_dir = skill_dir
        self.error_callback = error_callback
        self.success_callback = success_callback
        self.runnable_config = runnable_config or RunnableConfig()

    async def update_message(self, message: BaseMessage) -> None:
        """更新消息列表，替换同ID的消息."""
        self.messages.append(message)
        await self.on_message_updated()

    async def on_message_updated(self) -> None:
        """当消息更新时调用，保存当前状态到存储."""
        await self.storage.save_messages(self.messages)

    async def _compress_context_if_needed(self, current_agent: str, next_agent: str, reason: str):
        """当控制权交回关键节点时，对历史记录进行压缩.

        这里简单采用基于规则的压缩方式，即：当控制权从Agent转移会Router时，找到该Agent执行期间的所有消息并删除，仅保留两次transfer消息
        """
        if next_agent != "Router":
            return
        # 拿到最后两条消息，因为这是转移回来的，需要保留
        transfer_back_msg = []
        last_msg_idx = len(self.messages) - 1
        cnt = 0
        # 向前遍历，找到从 Router -> current_agent 的转移消息，保留该消息
        for i in range(len(self.messages)-1, -1, -1):
            cnt += 1
            msg = self.messages[i]
            if cnt <= 2:
                transfer_back_msg.append(msg)
            if isinstance(msg, ToolMessage) \
                and msg.content.startswith(f"TRANSFER_TO:{current_agent}"): # type: ignore
                last_msg_idx = i
                break
        # 保留从开始到last_msg_idx的消息，和最后一条转移消息，删除中间的消息
        self.messages = self.messages[:last_msg_idx+1]
        # 一条汇总消息，直接使用 reason 来说明当前Agent的执行结果，避免模型在生成汇总消息时出现问题
        task_summary = transfer_back_msg[1].content if len(transfer_back_msg) > 1 else ""
        task_summary = task_summary if task_summary else reason
        self.messages.append(AIMessage(content=task_summary))
        # 添加从 current_agent 转移回 Router 的消息，应当是两条，一个tool_calls，一个tool_result
        if transfer_back_msg and not reason.startswith('Maximum epochs reached'):
            # 如果不是因为达到最大轮次而被强制转移回 Router 的，就添加正常的转移消息；
            self.messages.extend(reversed(transfer_back_msg))
        else:
            # 因为达到最大轮次被强制转回Router，添加一条Human消息，说明强制转移的原因，避免模型在生成汇总消息时出现问题
            self.messages.append(HumanMessage(content=reason))
        await self.on_message_updated()

    async def handle_error(self, error_message: str):
        """处理错误，调用错误回调函数并传入当前消息列表和响应列表."""
        if self.error_callback:
            messages=[msg.model_dump() for msg in self.messages]
            response = await self.storage.load_response()
            if not response:
                response = []
            response.append({"resp_type": "error", "content": error_message})
            await self.error_callback(self,
                                      error_message,
                                        messages,
                                        response)

    async def handle_success(self, success_message: str):
        """处理成功，调用成功回调函数并传入当前消息列表和响应列表."""
        if self.success_callback:
            messages=[msg.model_dump() for msg in self.messages]
            response = await self.storage.load_response()
            if not response:
                response = []
            response.append({"resp_type": "success", "content": success_message})
            await self.success_callback(self,
                                        success_message,
                                        messages,
                                        response)

    async def process_message(self, user_input: str) -> AsyncGenerator[dict[str, Any], None]: # noqa
        if not user_input.strip():
            return

        if not self.messages:
            self.messages = await self.storage.load_messages()

        if not self.agents:
            self.agents = await self.agent_factory(self.tenant_id, self.session_id, self.skill_dir, self.runnable_config) # noqa
        # Add user message
        await self.update_message(HumanMessage(content=user_input))
        # Start processing loop until we need user input again
        # The loop continues as long as a transfer happens
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
            # Send status update
            yield {"resp_type": "status", "content": f"Agent {self.active_agent_name} is thinking..."}

            try:
                epoch += 1
                # Invoke agent
                # Note: Assuming invoke returns the full list of messages or state
                next_agent_name = None
                reason = "No reason provided"  # Default reason
                async for chunk in active_agent.astream({"messages": self.messages}): # type: ignore
                    await self.update_message(chunk)  # Update the message list with the new chunk
                    msg = chunk
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            yield {
                                "resp_type": "text",
                                "agent": self.active_agent_name,
                                "content": msg.content,
                            }
                        elif is_tool_calls(msg):
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call['name']
                                args = tool_call.get('args', {})
                                if not is_transfer_call(tool_call): # type: ignore
                                    yield {
                                        "resp_type": "tool_call",
                                        "agent": self.active_agent_name,
                                        "content": "called tool: " + tool_name + " with args: " + str(args),
                                        "tool_name": tool_name,
                                        "tool_args": args,
                                    }
                    if isinstance(msg, ToolMessage):
                        if msg.content.startswith("TRANSFER_TO:"): # type: ignore
                            content = msg.content.replace("TRANSFER_TO:", "") # type: ignore
                            parts = content.split(":")
                            if len(parts) >= 2:
                                next_agent_name = parts[0]
                                reason = ':'.join(parts[1:])
                                yield {
                                    "resp_type": "transfer",
                                    "from_agent": self.active_agent_name,
                                    "to_agent": next_agent_name,
                                    "reason": reason
                                }
                        else:
                            yield {
                                "resp_type": "tool_result",
                                "agent": self.active_agent_name,
                                "content": msg.content,
                            }

                if next_agent_name and next_agent_name in self.agents:
                    self.active_agent_name = next_agent_name
                    await self._compress_context_if_needed(active_agent.name, next_agent_name, reason)
                    await self.update_message(HumanMessage(content=f"You have been transferred control by {active_agent.name}. Reason: {reason}. Please continue to process the task.")) # noqa: E501
                    continue
                else:
                    if self.success_callback:
                        await self.handle_success("Task completed successfully.")
                    yield {"resp_type": "finished", "content": "Task processing completed."}
                    # Save the current state of messages back to storage
                    await self.storage.save_messages(self.messages)
                    break

            except Exception as e:
                import traceback
                traceback.print_exc()
                await self.handle_error(str(e))
                yield {"resp_type": "error", "content": str(e)}
                break
