import operator
from typing import Annotated, TypedDict


# Define the state of our multi-agent system
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str  # The name of the next agent to route to
    active_agent: str  # The name of the current active agent
