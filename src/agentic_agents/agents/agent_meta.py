
from pydantic import BaseModel


class AgentMeta(BaseModel):
    """Agent 元信息."""
    name: str
    duty: str
    skills: list[str]
    system_prompt: str
    mcp_servers: list[str] = []
    """该 Agent 可使用的 MCP 服务器名称列表"""


class MCPServerConfig(BaseModel):
    """MCP 服务器配置."""
    name: str
    transport: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    auth: str | None = None
    headers: dict[str, str] | None = None


class AgentConfig(BaseModel):
    """Agent 配置文件结构."""
    metadata: dict
    agents: list[AgentMeta]
    mcp_servers: list[MCPServerConfig] = []
