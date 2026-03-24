"""
Agent 工厂函数 - 用于构建多 Agent 系统。
"""

import logging
import os
from typing import Any

from .llm.base import LLMClient
from .llm.openai_client import OpenAIClient
from .llm.anthropic_client import AnthropicClient

from .agents.agent_meta import AgentConfig
from .agents.base import Agent
from .agents.handoff import create_transfer_tool
from .skills.loader import Skill, load_skills_from_directory
from .skills.meta_tools import lookup_skill_tool, set_global_skills
from .tools.base import AgentTool
from .tools.basic_tools import current_time_tool

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


async def build_agent(
    name: str,
    duty: str,
    skills: list[Skill],
    llm: LLMClient,
    prompt: str,
    tenant_id: str = "",
    session_id: str = "",
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
    """
    tools: list[AgentTool] = []

    # 从 skills 中收集工具
    for skill in skills:
        tools.extend(skill.tools)

    # 添加内置工具
    tools.append(lookup_skill_tool)
    tools.append(create_transfer_tool(
        "Router",
        "当完成了你当前的任务，或者无法继续完成任务时，使用该工具将控制权交还给 Router。并简要说明交还控制权的原因。",
    ))
    tools.append(current_time_tool)

    skill_desc = [f"- {s.name}: {s.description}" for s in skills]
    skill_list = "\n\t".join(skill_desc)

    system_prompt = f"""You are {name}
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
"""

    return Agent(
        name=name,
        description=duty,
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


async def build_agents(
    tenant_id: str,
    session_id: str,
    skills_dir: str,
    llm: LLMClient | None = None,
    agent_config_path: str = "agents.json",
) -> dict[str, Agent]:
    """构建所有 Agents 的工厂函数，返回一个字典映射 agent_name -> Agent 实例.

    Args:
        tenant_id: 租户 ID
        session_id: 会话 ID
        skills_dir: 技能目录路径
        llm: LLM 客户端，不提供则使用默认的 OpenAI 客户端
        agent_config_path: Agent 配置文件路径
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

    if not os.path.exists(agent_config_path):
        logger.warning(f"Agent config file not found: {agent_config_path}")
        agent_config = AgentConfig(metadata={}, agents=[])
    else:
        agent_config_content = open(agent_config_path).read()
        logger.info(f"Loaded agent configuration: {agent_config_content}")
        agent_config = AgentConfig.model_validate_json(agent_config_content)

    router_tools: list[AgentTool] = []

    for agent_meta in agent_config.agents:
        skills = [skill_map.get(skill_name) for skill_name in agent_meta.skills]
        skills = [s for s in skills if s is not None]
        agents[agent_meta.name] = await build_agent(
            name=agent_meta.name,
            duty=agent_meta.duty,
            skills=skills,
            llm=llm,
            prompt=agent_meta.system_prompt,
            tenant_id=tenant_id,
            session_id=session_id,
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
你的职责不是直接回答问题，而是理解用户的意图，并将任务分发给最合适的专家代理。

可用代理:
{chr(10).join([f"- {a.name}: {a.description} \n" for a in agents.values() if a.name != "Router"])}

请使用对应的 transfer 工具进行转发。
如果用户只是打招呼或闲聊，你可以直接回复。
""",
    )
    return agents
