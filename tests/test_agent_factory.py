"""agent_factory 模块单元测试 - build_agent / build_agents."""

import textwrap

import pytest
from unittest.mock import AsyncMock, patch

from agentic_agents.agent_factory import build_agent, build_agents
from agentic_agents.agents.agent_meta import AgentConfig, AgentMeta
from agentic_agents.agents.base import Agent
from agentic_agents.llm.base import LLMClient
from agentic_agents.skills.loader import Skill
from agentic_agents.tools.base import AgentTool


def _mock_llm() -> AsyncMock:
    llm = AsyncMock(spec=LLMClient)
    llm.supports_tools.return_value = True
    return llm


def _make_skill(name: str = "TestSkill") -> Skill:
    tool = AgentTool(
        name="skill_tool",
        description="A skill tool",
        func=lambda: "ok",
    )
    return Skill(
        name=name,
        description=f"{name} description",
        instruction="Use this skill.",
        tools=[tool],
        path="/tmp",
    )


# ---------------------------------------------------------------------------
# build_agent
# ---------------------------------------------------------------------------

class TestBuildAgent:
    @pytest.mark.asyncio
    async def test_returns_agent(self):
        llm = _mock_llm()
        agent = await build_agent(
            name="Worker",
            duty="Do work",
            skills=[_make_skill()],
            llm=llm,
            prompt="You do work.",
            session_id="s1",
        )
        assert isinstance(agent, Agent)
        assert agent.name == "Worker"

    @pytest.mark.asyncio
    async def test_has_builtin_tools(self):
        llm = _mock_llm()
        agent = await build_agent(
            name="Worker",
            duty="Do work",
            skills=[],
            llm=llm,
            prompt="prompt",
        )
        tool_names = [t.name for t in agent.tools]
        assert "lookup_skill" in tool_names
        assert "transfer_to_Router" in tool_names
        assert "current_time" in tool_names

    @pytest.mark.asyncio
    async def test_system_prompt_contains_agent_name(self):
        llm = _mock_llm()
        agent = await build_agent(
            name="SearchAgent",
            duty="Search",
            skills=[_make_skill("SearchSkill")],
            llm=llm,
            prompt="Search the web",
            session_id="s1",
        )
        assert "SearchAgent" in agent.system_prompt
        assert "SearchSkill" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_mcp_tools_included(self):
        llm = _mock_llm()
        mcp_tool = AgentTool(
            name="mcp_read_file",
            description="Read file via MCP",
            func=lambda: "content",
        )
        agent = await build_agent(
            name="Worker",
            duty="Do work",
            skills=[],
            llm=llm,
            prompt="prompt",
            mcp_tools=[mcp_tool],
        )
        tool_names = [t.name for t in agent.tools]
        assert "mcp_read_file" in tool_names


# ---------------------------------------------------------------------------
# build_agents
# ---------------------------------------------------------------------------

class TestBuildAgents:
    @pytest.mark.asyncio
    async def test_builds_all_agents_plus_router(self, tmp_path):
        # 创建空的 skills 目录
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = AgentConfig(
            metadata={},
            agents=[
                AgentMeta(
                    name="SearchAgent",
                    duty="Search the web",
                    skills=[],
                    system_prompt="You search the web.",
                ),
                AgentMeta(
                    name="CodeAgent",
                    duty="Write code",
                    skills=[],
                    system_prompt="You write code.",
                ),
            ],
        )

        llm = _mock_llm()
        agents = await build_agents(config, "s1", str(skills_dir), llm=llm)

        assert "Router" in agents
        assert "SearchAgent" in agents
        assert "CodeAgent" in agents
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_router_has_transfer_tools(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = AgentConfig(
            metadata={},
            agents=[
                AgentMeta(
                    name="Worker",
                    duty="Do work",
                    skills=[],
                    system_prompt="prompt",
                ),
            ],
        )

        llm = _mock_llm()
        agents = await build_agents(config, "s1", str(skills_dir), llm=llm)

        router = agents["Router"]
        tool_names = [t.name for t in router.tools]
        assert "transfer_to_Worker" in tool_names

    @pytest.mark.asyncio
    async def test_skills_loaded_for_agent(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skill_sub = skills_dir / "my_skill"
        skill_sub.mkdir(parents=True)

        (skill_sub / "skill.md").write_text(textwrap.dedent("""\
            ---
            name: MySkill
            description: My skill
            ---
            Instructions here.
        """))

        (skill_sub / "tools.py").write_text(textwrap.dedent("""\
            from agentic_agents.tools.base import create_tool
            my_tool = create_tool(
                name="my_tool",
                description="d",
                parameters={"type": "object", "properties": {}},
                func=lambda: "ok",
            )
        """))

        config = AgentConfig(
            metadata={},
            agents=[
                AgentMeta(
                    name="Worker",
                    duty="Do work",
                    skills=["MySkill"],
                    system_prompt="prompt",
                ),
            ],
        )

        llm = _mock_llm()
        agents = await build_agents(config, "s1", str(skills_dir), llm=llm)

        worker = agents["Worker"]
        assert "MySkill" in worker.system_prompt
