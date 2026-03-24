
from langchain_core.tools import tool


# This tool will be returned to the main loop to signal a transfer
class Handoff:
    """A signal to transfer control to another agent."""
    def __init__(self, target_agent: str, reason: str = ""):
        self.target_agent = target_agent
        self.reason = reason

def create_transfer_tool(target_agent_name: str, description: str):
    """Creates a tool that an agent can use to transfer control to another specific agent."""

    # Using return_direct=True ensures the agent execution stops immediately after
    # calling this tool, and returns the output to the outer loop.
    # This prevents the agent from running in a circle or trying to add more context.
    @tool(f"transfer_to_{target_agent_name}",
          return_direct=True,)
    def transfer(reason: str = "") -> str:
        """Transfers control to the {target_agent_name}.

        Use this when the user's request is better handled by {target_agent_name}.

        Args:
            reason: The reason for transferring (context for the next agent, what to do and what have done).
        """
        # We return a special string or object that the runtime loop detects
        return f"TRANSFER_TO:{target_agent_name}:{reason}"

    # Update docstring dynamically
    transfer.__doc__ = transfer.__doc__.format(target_agent_name=target_agent_name)

    return transfer
