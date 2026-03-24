from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .helpers import is_tool_calls, is_transfer_call


class Agent:
    def __init__(
        self,
        name: str,
        description: str,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str,
        max_epochs: int = 50,
        runnable_config: RunnableConfig | None = None,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        # Create the underlying LangGraph agent
        self.executor = create_agent(llm, tools)
        self.max_epochs = max_epochs
        self.runnable_config = runnable_config or RunnableConfig()

    async def astream(self, state):  # noqa
        # We manually prepend the system prompt.
        input_messages = state["messages"]
        invocation_messages = [SystemMessage(content=self.system_prompt)] + input_messages
        try:
            # We look for the moment a ToolMessage with 'TRANSFER_TO' is produced.
            in_epoch = 0
            async for chunk in self.executor.astream(
                {"messages": invocation_messages}, config=self.runnable_config, stream_mode="messages"
            ):
                if isinstance(chunk, tuple):
                    current_messages = [chunk[0]]
                elif "model" in chunk:
                    current_messages = chunk["model"].get("messages", [])
                else:
                    current_messages = chunk.get("messages", [])
                in_epoch += 1
                if in_epoch > self.max_epochs:
                    print('-'*100)
                    print(f"DEBUG {self.name}: Maximum epochs reached, forcing transfer.")
                    yield ToolMessage(content=f"TRANSFER_TO:Router:Maximum epochs reached in agent {self.name}",
                                    tool_call_id="")  # type: ignore
                    return
                for msg in current_messages:
                    invocation_messages.append(msg)
                    yield msg
                    if is_tool_calls(msg):
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call['name']
                            args = tool_call.get('args', {})
                            tool = next((t for t in self.tools if t.name == tool_name), None)
                            call_id = tool_call.get('id', '')
                            if is_transfer_call(tool_call): # type: ignore
                                # 如果是转移调用，直接返回转移指令，不执行工具
                                if tool:
                                    result = tool.run(args)
                                    yield ToolMessage(content=result, tool_call_id=call_id)  # type: ignore
                                    return
                                else:
                                    # 不能转移到未知的Agent
                                    print('-'*100)
                                    print(f"DEBUG {self.name}: Transfer tool call detected but target agent not found, continuing execution.")

        except Exception as e:
            print('-'*100)
            print(f"DEBUG {self.name}: ERROR in stream: {e}")
            raise e

    async def invoke(self, state):  # noqa
        final_state = {"messages": state.get("messages", [])}
        async for chunk in self.astream(state): # type: ignore
            final_state = chunk
        return final_state
