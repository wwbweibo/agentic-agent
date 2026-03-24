from langchain.messages import AIMessage


def is_tool_calls(message: AIMessage) -> bool:
    """检查消息中是否包含工具调用."""
    return isinstance(message, AIMessage) and len(message.tool_calls) > 0

def is_transfer_call(tool_call: dict) -> bool: # noqa
    return tool_call.get('name', '').startswith('transfer_to_')

def transfer_args(tool_call: dict) -> tuple[str, str]: # noqa
    tool_name = tool_call.get('name', '')
    target = tool_name.split("transfer_to_")[1]
    args = tool_call.get('args', {})
    reason = args.get('reason', 'No reason provided')
    next_agent_name = target.strip()
    return next_agent_name, reason

def is_lookup_skill(tool_call: dict) -> bool: # noqa
    """检查工具调用是否是技能查询."""
    return tool_call.get('name', '') == 'lookup_skill'
