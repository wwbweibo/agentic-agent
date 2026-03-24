"""
辅助函数 - 用于处理工具调用和消息类型判断。
"""


def is_transfer_call(tool_name: str) -> bool:
    """判断工具名称是否是转移调用."""
    return tool_name.startswith("transfer_to_")


def extract_transfer_target(tool_name: str, arguments: dict) -> tuple[str, str]:
    """从转移工具调用中提取目标 agent 和原因.

    Args:
        tool_name: 工具名称，格式为 transfer_to_<agent_name>
        arguments: 工具参数
    """
    target = tool_name.split("transfer_to_", 1)[1].strip()
    reason = arguments.get("reason", "No reason provided")
    return target, reason
