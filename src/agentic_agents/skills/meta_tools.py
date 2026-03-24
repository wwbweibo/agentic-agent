from .loader import Skill
from ..tools.base import AgentTool, create_tool

# 全局存储已加载的 Skills，以便 lookup_skill 工具使用
GLOBAL_SKILLS: dict[str, Skill] = {}


def set_global_skills(skills: list[Skill]) -> None:
    global GLOBAL_SKILLS
    GLOBAL_SKILLS = {s.name: s for s in skills}


lookup_skill_tool = create_tool(
    name="lookup_skill",
    description="使用该工具来获取技能具体说明。当你需要使用某个技能时，首先使用该工具来查询该技能的详细使用说明和能力范围，以确保你正确使用它。",
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "技能名称",
            },
        },
        "required": ["skill_name"],
    },
    func=lambda skill_name: _do_lookup(skill_name),
)


def _do_lookup(skill_name: str) -> str:
    skill = GLOBAL_SKILLS.get(skill_name)
    if not skill:
        return f"Error: Skill '{skill_name}' not found."

    return f"""
*** Skill Guide: {skill.name} ***
Description: {skill.description}

### Detailed Instructions:
{skill.instruction}

### Available Tools:
{', '.join([t.name for t in skill.tools])}
"""
