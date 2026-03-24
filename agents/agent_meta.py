from openai import BaseModel


class AgentMeta(BaseModel):
    """Agent 元信息."""
    name: str
    duty: str
    skills: list[str]
    system_prompt: str

class AgentConfig(BaseModel):
    metadata: dict
    agents: list[AgentMeta]

class MCPServer(BaseModel):
    """MCP Server 元信息."""
    name: str
    transport: str
    connection_info: dict
