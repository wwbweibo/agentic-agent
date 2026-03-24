from pydantic import BaseModel


class AgentMeta(BaseModel):
    """Agent 元信息."""
    name: str
    duty: str
    skills: list[str]
    system_prompt: str


class AgentConfig(BaseModel):
    """Agent 配置文件结构."""
    metadata: dict
    agents: list[AgentMeta]
