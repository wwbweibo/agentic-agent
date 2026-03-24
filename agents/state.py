import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


# Define the state of our multi-agent system
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str # The name of the next agent to route to
    active_agent: str # The name of the current active agent
