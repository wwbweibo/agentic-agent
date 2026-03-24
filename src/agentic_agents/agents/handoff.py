from ..tools.base import AgentTool, create_tool


def create_transfer_tool(target_agent_name: str, description: str) -> AgentTool:
    """创建一个用于控制权转移的工具.

    使用 return_direct=True 等价的行为：工具执行后立即返回，
    由外层 AgentSession 检测到 transfer 事件后中断循环。
    """

    def transfer(reason: str = "") -> str:
        """转移控制权给 {target_agent_name}.

        当用户的请求更适合由 {target_agent_name} 处理时使用此工具。

        Args:
            reason: 转移原因（传递给下一个 agent 的上下文）
        """
        return f"TRANSFER_TO:{target_agent_name}:{reason}"

    return create_tool(
        name=f"transfer_to_{target_agent_name}",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "转移原因（传递给下一个 agent 的上下文）",
                },
            },
            "required": [],
        },
        func=transfer,
    )
