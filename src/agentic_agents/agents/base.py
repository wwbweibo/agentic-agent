import json
from typing import Any

from ..llm.base import AgentMessage, ChatResult, LLMClient, ToolCall
from ..tools.base import AgentTool


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: LLMClient,
        tools: list[AgentTool],
        system_prompt: str,
        max_epochs: int = 50,
    ):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_epochs = max_epochs

    def _tool_definitions(self) -> list[dict]:
        """将工具列表转换为 LLM 工具定义格式."""
        return [tool.to_dict() for tool in self.tools]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """执行单个工具调用."""
        tool = next((t for t in self.tools if t.name == tool_call.name), None)
        if not tool:
            return f"Error: Tool '{tool_call.name}' not found."
        try:
            result = tool.execute(**tool_call.arguments)
            # 支持 async 工具
            import asyncio
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_call.name}': {e}"

    def _is_transfer_call(self, tool_call: ToolCall) -> bool:
        """判断工具调用是否是转移调用."""
        return tool_call.name.startswith("transfer_to_")

    def _extract_transfer_target(self, tool_call: ToolCall) -> tuple[str, str]:
        """从转移工具调用中提取目标 agent 和原因."""
        # 格式: transfer_to_<agent_name>
        target = tool_call.name.split("transfer_to_", 1)[1].strip()
        reason = tool_call.arguments.get("reason", "No reason provided")
        return target, reason

    async def astream(self, state: dict) -> Any:
        """异步流式执行 Agent.

        使用自己的 agent loop 替代 langchain.agents.create_agent。
        循环调用 LLM 直到完成或遇到 transfer 工具调用。
        """
        input_messages: list[dict] = state.get("messages", [])

        # 构建包含 system prompt 的消息列表
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]
        messages.extend(input_messages)

        tool_definitions = self._tool_definitions()

        in_epoch = 0
        while True:
            in_epoch += 1
            if in_epoch > self.max_epochs:
                print("-" * 100)
                print(f"DEBUG {self.name}: Maximum epochs reached, forcing transfer.")
                yield {
                    "type": "transfer",
                    "from_agent": self.name,
                    "to_agent": "Router",
                    "reason": f"Maximum epochs reached in agent {self.name}",
                    "tool_result": f"TRANSFER_TO:Router:Maximum epochs reached in agent {self.name}",
                }
                return

            # 调用 LLM
            try:
                result: ChatResult = await self.llm.chat(
                    messages=messages,
                    tools=tool_definitions if tool_definitions else None,
                )
            except Exception as e:
                print("-" * 100)
                print(f"DEBUG {self.name}: LLM call error: {e}")
                raise e

            msg = result.message

            # 如果有文本内容，产出文本
            if msg.content:
                yield {
                    "type": "text",
                    "agent": self.name,
                    "content": msg.content,
                }

            # 如果有工具调用
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 将 assistant 的 tool_call 添加到消息历史
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                },
                            }
                        ],
                    })

                    if self._is_transfer_call(tool_call):
                        # 转移调用：直接执行并返回 transfer 事件
                        target, reason = self._extract_transfer_target(tool_call)
                        tool_result_str = f"TRANSFER_TO:{target}:{reason}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_str,
                        })
                        yield {
                            "type": "transfer",
                            "from_agent": self.name,
                            "to_agent": target,
                            "reason": reason,
                            "tool_result": tool_result_str,
                        }
                        return
                    else:
                        # 普通工具调用：执行并添加结果
                        tool_result_str = self._execute_tool(tool_call)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_str,
                        })
                        yield {
                            "type": "tool_result",
                            "agent": self.name,
                            "tool_name": tool_call.name,
                            "tool_id": tool_call.id,
                            "content": tool_result_str,
                        }
            else:
                # 无工具调用：正常结束
                return

    async def invoke(self, state: dict) -> dict:
        """同步调用接口，收集所有事件并返回最终状态."""
        events = []
        async for event in self.astream(state):
            events.append(event)
        return {"events": events}
