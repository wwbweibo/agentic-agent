"""
Agent 工厂函数 - 用于构建多 Agent 系统。
"""

import logging
import os
from typing import Any

from .agents.agent_meta import AgentConfig, MCPServerConfig
from .agents.base import Agent
from .agents.handoff import create_transfer_tool
from .llm import AnthropicClient, OpenAIClient
from .llm.base import LLMClient
from .mcp import connect_mcp_server, get_mcp_client, disconnect_mcp_server
from .mcp.tools import mcp_tools_to_agent_tools
from .skills.loader import Skill, load_skills_from_directory
from .skills.meta_tools import lookup_skill, set_global_skills
from .tools.base import AgentTool
from .tools.basic_tools import current_time

logger = logging.getLogger(__name__)


def _make_llm(provider: str = "openai", **kwargs) -> LLMClient:
    """创建 LLM 客户端.

    Args:
        provider: "openai" 或 "anthropic"
        **kwargs: 传递给客户端的参数
    """
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    return OpenAIClient(**kwargs)


async def _connect_mcp_servers(
    servers: list[MCPServerConfig],
) -> dict[str, list[AgentTool]]:
    """连接 MCP 服务器并返回工具映射。

    Args:
        servers: MCP 服务器配置列表

    Returns:
        服务器名称到工具列表的映射
    """
    server_tools: dict[str, list[AgentTool]] = {}

    for server in servers:
        logger.info(f"Connecting MCP server: {server.name} (transport={server.transport})")
        client = get_mcp_client(server.name)  # 检查是否已连接
        if client:
            try:
                logger.info(f"MCP server '{server.name}' is already connected, skipping...")
                mcp_tools = await client.list_tools()
                server_tools[server.name] = mcp_tools_to_agent_tools(mcp_tools, client)
                continue
            except Exception as e:
                logger.error(f"Failed to list tools from already connected MCP server '{server.name}': {e}")
                # 如果连接有问题，继续尝试重新连接
                await disconnect_mcp_server(server.name)
        tools = await _connect_single_mcp_server(server)
        server_tools[server.name] = tools

    return server_tools

async def _connect_single_mcp_server(server: MCPServerConfig) -> list[AgentTool]:
    kwargs: dict[str, Any] = {}
    if server.transport == "stdio":
        kwargs["command"] = server.command
        if server.args:
            kwargs["args"] = server.args
        if server.env:
            kwargs["env"] = server.env
        if server.cwd:
            kwargs["cwd"] = server.cwd
    elif server.transport in ("http", "sse"):
        kwargs["url"] = server.url
        if server.auth:
            kwargs["auth"] = server.auth
        if server.headers:
            kwargs["headers"] = server.headers
    else:
        logger.warning(f"Unknown MCP transport: {server.transport}")
        return []
    try:
        tools = await connect_mcp_server(
            name=server.name,
            transport=server.transport,
            **kwargs,
        )
        return tools
    except Exception as e:
        logger.error(f"Failed to connect MCP server '{server.name}': {e}")
        return []

async def build_agent(
    name: str,
    duty: str,
    skills: list[Skill],
    llm: LLMClient,
    prompt: str,
    session_id: str = "",
    mcp_tools: list[AgentTool] | None = None,
) -> Agent:
    """Build a single agent with given skills and llm.

    Args:
        name: Agent name
        duty: Agent duty description
        skills: List of skills the agent has
        llm: The language model client to use
        prompt: System prompt for the agent
        tenant_id: Current tenant ID, can be used in the prompt for data scoping
        session_id: Current session ID, can be used in the prompt for data scoping
        mcp_tools: MCP tools available to this agent
    """
    tools: list[AgentTool] = list(mcp_tools) if mcp_tools else []

    # 添加内置工具
    tools.append(lookup_skill)
    tools.append(create_transfer_tool(
        "Router",
        "当完成了你当前的任务，或者无法继续完成任务时，使用该工具将控制权交还给 Router。并简要说明交还控制权的原因。",
    ))
    tools.append(current_time)

    skill_desc = [f"- {s.name}: {s.description}" for s in skills]
    skill_list = "\n\t".join(skill_desc)

    system_prompt = f"""你是一个专业的任务执行代理，名为 {name}。

【你的职责】
{prompt}

【工作流程】
1. 理解用户的需求
2. 使用 ReAct 模式完成任务：思考(Thought) -> 行动(Action) -> 观察(Observation)
3. 完成任务后，将结果返回给用户

【技能说明】
你有以下可用的技能和工具：
{skill_list}

使用 `lookup_skill` 工具可以查看技能的详细说明和使用方法。

【工具说明】
可用的外部工具：
{chr(10).join([f"- {t.name}: {t.description}" for t in tools if t.name != 'lookup_skill'])}

【重要规则】
1. 用中文回复
2. 不要编造信息，不知道的就说不知道
3. 当前会话 ID: {session_id}

【任务完成标准】
当你满足以下任一条件时，调用 `transfer_to_Router` 工具返回控制权：
- 任务已完成，能给出明确的答案或结果
- 任务超出你的能力范围
- 遇到错误无法继续

在 reason 参数中简要说明：
- 已完成的任务：给出答案摘要
- 未完成的任务：说明原因和已尝试的方法
"""

    return Agent(
        name=name,
        description=duty,
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


async def build_agents(
    agent_config: AgentConfig,
    session_id: str,
    skills_dir: str,
    llm: LLMClient | None = None,
) -> dict[str, Agent]:
    """构建所有 Agents 的工厂函数，返回一个字典映射 agent_name -> Agent 实例.

    Args:
        session_id: 会话 ID
        skills_dir: 技能目录路径
        llm: LLM 客户端，不提供则使用默认的 OpenAI 客户端
    """
    all_skills = load_skills_from_directory(skills_dir)
    set_global_skills(all_skills)
    skill_map = {s.name: s for s in all_skills}
    agents: dict[str, Agent] = {}

    if llm is None:
        llm = OpenAIClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    # 连接 MCP 服务器
    server_tools: dict[str, list[AgentTool]] = {}
    if agent_config.mcp_servers:
        logger.info(f"Connecting {len(agent_config.mcp_servers)} MCP servers...")
        server_tools = await _connect_mcp_servers(agent_config.mcp_servers)

    router_tools: list[AgentTool] = []

    for agent_meta in agent_config.agents:
        skills = [skill_map.get(skill_name) for skill_name in agent_meta.skills]
        skills = [s for s in skills if s is not None]

        # 收集该 Agent 关联的 MCP 工具
        mcp_tools: list[AgentTool] = []
        for server_name in agent_meta.mcp_servers:
            if server_name in server_tools:
                mcp_tools.extend(server_tools[server_name])

        agents[agent_meta.name] = await build_agent(
            name=agent_meta.name,
            duty=agent_meta.duty,
            skills=skills,
            llm=llm,
            prompt=agent_meta.system_prompt,
            session_id=session_id,
            mcp_tools=mcp_tools,
        )
        router_tools.append(create_transfer_tool(
            agent_meta.name,
            f"当用户提出、询问与{agent_meta.duty}相关问题或者任务时，调用该工具将控制权转移给{agent_meta.name}。",
        ))

    agents["Router"] = Agent(
        name="Router",
        description="总管代理，负责接待用户并分发任务。",
        llm=llm,
        tools=router_tools,
        system_prompt=f"""你是一个多智能体系统的接待员 (Router)。
你的职责是理解用户意图，将任务分发给最合适的专家代理，并汇总结果回复用户。

可用代理:
{chr(10).join([f"- {a.name}: {a.description} \n" for a in agents.values() if a.name != "Router"])}

【重要】指令：
1. 如果用户只是打招呼或闲聊，可以直接回复。
2. 如果需要转移任务，使用 transfer 工具。
3. 当其他 Agent 完成任务并将控制权交还给你时：
   - 检查消息历史中的内容
   - 如果任务已完成，直接用中文回复用户最终结果
   - 不要再次尝试分发已经完成的任务！
4. 你的回复将直接呈现给用户，请确保回复完整、友好。

回复格式：直接给出答案或结论，不需要额外说明你做了什么。
""",
    )
    return agents
