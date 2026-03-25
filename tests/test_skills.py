"""skills 模块单元测试 - Skill 加载和 lookup_skill."""

import os
import textwrap

import pytest

from agentic_agents.skills.loader import Skill, load_skills_from_directory
from agentic_agents.skills.meta_tools import (
    GLOBAL_SKILLS,
    _do_lookup,
    lookup_skill,
    set_global_skills,
)
from agentic_agents.tools.base import AgentTool


# ---------------------------------------------------------------------------
# load_skills_from_directory
# ---------------------------------------------------------------------------

class TestLoadSkills:
    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = load_skills_from_directory(str(tmp_path / "nonexistent"))
        assert result == []

    def test_load_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()

        # skill.md
        (skill_dir / "skill.md").write_text(textwrap.dedent("""\
            ---
            name: MySkill
            description: A test skill
            ---
            Use this skill to do things.
        """))

        # tools.py
        (skill_dir / "tools.py").write_text(textwrap.dedent("""\
            from agentic_agents.tools.base import create_tool

            my_tool = create_tool(
                name="my_tool",
                description="A tool",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                func=lambda q: f"result:{q}",
            )
        """))

        skills = load_skills_from_directory(str(tmp_path))
        assert len(skills) == 1
        assert skills[0].name == "MySkill"
        assert skills[0].description == "A test skill"
        assert len(skills[0].tools) == 1
        assert skills[0].tools[0].name == "my_tool"

    def test_skip_non_directory(self, tmp_path):
        (tmp_path / "not_a_dir.txt").write_text("hello")
        result = load_skills_from_directory(str(tmp_path))
        assert result == []

    def test_skip_dir_missing_files(self, tmp_path):
        (tmp_path / "incomplete_skill").mkdir()
        (tmp_path / "incomplete_skill" / "skill.md").write_text("---\nname: X\n---\n")
        # 没有 tools.py
        result = load_skills_from_directory(str(tmp_path))
        assert result == []

    def test_tools_get_tagged(self, tmp_path):
        skill_dir = tmp_path / "tagged_skill"
        skill_dir.mkdir()

        (skill_dir / "skill.md").write_text(textwrap.dedent("""\
            ---
            name: TaggedSkill
            description: Tagged
            ---
            Instructions.
        """))

        (skill_dir / "tools.py").write_text(textwrap.dedent("""\
            from agentic_agents.tools.base import create_tool
            t = create_tool(
                name="t",
                description="d",
                parameters={"type": "object", "properties": {}},
                func=lambda: "ok",
            )
        """))

        skills = load_skills_from_directory(str(tmp_path))
        assert len(skills) == 1
        assert any("TaggedSkill" in tag for tag in skills[0].tools[0].tags)


# ---------------------------------------------------------------------------
# meta_tools: set_global_skills / lookup_skill
# ---------------------------------------------------------------------------

class TestMetaTools:
    def setup_method(self):
        """每个测试前清理全局状态."""
        GLOBAL_SKILLS.clear()

    def test_set_global_skills(self):
        skill = Skill(
            name="TestSkill",
            description="A test",
            instruction="Do things",
            tools=[],
            path="/tmp",
        )
        set_global_skills([skill])
        from agentic_agents.skills.meta_tools import GLOBAL_SKILLS as current
        assert "TestSkill" in current

    def test_lookup_existing_skill(self):
        tool = AgentTool(name="t", description="d", func=lambda: None)
        skill = Skill(
            name="SearchSkill",
            description="Search things",
            instruction="Use web_search to find info.",
            tools=[tool],
            path="/tmp",
        )
        set_global_skills([skill])

        result_text, result_tools = _do_lookup("SearchSkill")
        assert "SearchSkill" in result_text
        assert "Search things" in result_text
        assert len(result_tools) == 1
        assert result_tools[0].name == "t"

    def test_lookup_nonexistent_skill(self):
        result_text, result_tools = _do_lookup("Missing")
        assert "not found" in result_text
        assert result_tools == []

    def test_lookup_skill_is_agent_tool(self):
        assert isinstance(lookup_skill, AgentTool)
        assert lookup_skill.name == "lookup_skill"
