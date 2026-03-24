import logging

from langchain.chat_models import BaseChatModel
from langchain_community.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from .agents.agent_meta import AgentConfig, MCPServer
from .agents.base import Agent
from .agents.handoff import create_transfer_tool
from .skills.loader import Skill, load_skills_from_directory
from .skills.meta_tools import lookup_skill, set_global_skills
from .tools.basic_tools import current_time

logger = logging.getLogger(__name__)

async def build_agent(name: str,
                      duty: str,
                      skills: list[Skill],
                      llm: BaseChatModel,
                      prompt: str,
                      tenant_id: str = "",
                      session_id: str = "",
                      mcp_server: list[MCPServer] | None= None,
                      runnable_config: RunnableConfig | None = None) -> Agent:
    """Build a single agent with given skills and llm.

    Args:
        name: Agent name
        duty: Agent duty description
        skills: List of skills the agent has
        llm: The language model to use
        prompt: System prompt for the agent
        tenant_id: Current tenant ID, can be used in the prompt for data scoping
        session_id: Current session ID, can be used in the prompt for data scoping
        mcp_server: List of MCP servers to connect for tool access
        runnable_config: Optional RunnableConfig for the agent's execution, if not provided, a default one will be used
    """
    if mcp_server is None:
        mcp_server = []
    tools: list[BaseTool] = []
    for skill in skills:
        tools.extend(skill.tools)

    if mcp_server:
        connects = {}
        for server in mcp_server:
            config = {
                "transport": server.transport,
            }
            if server.transport == "http":
                config['url'] = server.connection_info['url']
            elif server.transport == "stdio":
                config['cmd'] = server.connection_info['cmd']
                config['args'] = server.connection_info.get('args', [])
                config['env'] = server.connection_info.get('env', {})
            connects[server.name] = config
        client = MultiServerMCPClient(connects)
        for tool in await client.get_tools():
            tools.append(tool)
    tools.append(lookup_skill)
    tools.append(create_transfer_tool("Router",
                    "当完成了你当前的任务，或者无法继续完成任务时，使用该工具将控制权交还给 Router。并简要说明交还控制权的原因。")) # type: ignore # noqa
    tools.append(current_time)

    skill_desc = [f"- {s.name}: {s.description}" for s in skills]
    skill_list = "\n\t".join(skill_desc)

    agent = Agent(
        name=name,
        description=duty,
        llm=llm,
        tools=tools,
        system_prompt=f"""You are {name}
{prompt}
your capabilities are determined by the skills and tools you have, you can not do anything beyond your skills.
Each skill has its own specific instructions and available tools.

You got the following skills and tools, please use them to help you complete the task:
{skill_list}

Please read carefully the instruction of each skill and use the tools provided by the skill using `lookup_skill` tool.
For External Tools, you can use them directly.

you have these tools available:
{chr(10).join([f"- {t.name}: {t.description}, tags: {t.tags}" for t in tools])}

【Important】: you can see multiple tools,
but only tools without tags are avaliable before you use `lookup_skill` to check the instructions of the skill you have.
so please make sure to use `lookup_skill` to check the instruction of the skill before using other tools.

If you encounter a task that is beyond your capabilities,
use `transfer_to_Router` tool to transfer control back to Router.
Router will reassign the task to other agents or handle it by itself.

When you complete the task, call `transfer_to_Router` tool to transfer control back to Router and report the result to Router in the reason parameter.
**Important**: You can ONLY Transfer control to Router. You CANNOT transfer control to any other agent, because you don't know who are the other agents, and transferring to unknown agents may cause loss of control.

Please flow the instructions blow:
1. Please make sure that all your answers are in 「Chinese」.
2. Do not make up any information. If you don't know, say you don't know. If there are no data, say no data.
3. Current session_id is {session_id}. You can use this session_id to query data within the current session scope.
4. Current tenant_id is {tenant_id}. You can use this tenant_id to query data within the current tenant scope.

You MUST follow the react pattern to complete the task.
The react pattern is a loop of "Thought -> Action -> Observation",
which can help you to complete the task step by step and check the result of each step.
""",
    runnable_config=runnable_config
)
    return agent


async def build_agents(tenant_id: str, session_id: str, skills_dir: str, runnable_config: RunnableConfig,
                       llm: BaseChatModel | None = None, agent_config_path: str = "agents.json") -> dict[str, Agent]:
    """构建所有 Agents 的工厂函数，返回一个字典映射 agent_name -> Agent 实例."""
    all_skills = load_skills_from_directory(skills_dir)
    set_global_skills(all_skills)  # 将技能加载到全局，以便工具使用
    skill_map = {s.name: s for s in all_skills}
    agents: dict[str, Agent] = {}

    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4",
            streaming=False
        )
    agent_config_content = open(agent_config_path).read()
    logger.info(f"Loaded agent configuration: {agent_config_content}")
    agent_config = AgentConfig.model_validate_json(agent_config_content)
    logger.info(f"Parsed agent configuration: {agent_config}")

    router_tools = []

    for agent in agent_config.agents:
        agents[agent.name] = await build_agent(
            agent.name,
            agent.duty,
            [skill_map.get(skill_name) for skill_name in agent.skills], # type: ignore
            llm,
            agent.system_prompt,
            tenant_id=tenant_id,
            session_id=session_id)
        router_tools.append(create_transfer_tool(
            agent.name,
            f"当用户提出、询问与{agent.duty}相关问题或者任务时，调用该工具将控制权转移给{agent.name}。")) # type: ignore

    agents["Router"] = Agent(
        name="Router",
        description="总管代理，负责接待用户并分发任务。",
        llm=llm,
        tools=router_tools, # type: ignore
        system_prompt=f"""你是一个多智能体系统的接待员 (Router)。
你的职责不是直接回答问题，而是理解用户的意图，并将任务分发给最合适的专家代理。

可用代理:
{chr(10).join([f"- {agent.name}: {agent.description} \n" for agent in agents.values() if agent.name != "Router"])}

请使用对应的 transfer 工具进行转发。
如果用户只是打招呼或闲聊，你可以直接回复。
"""
    )
    return agents
