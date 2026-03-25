from .loader import Skill
from ..tools.base import tool, AgentTool

# 全局存储已加载的 Skills，以便 lookup_skill 工具使用
GLOBAL_SKILLS: dict[str, Skill] = {}


def set_global_skills(skills: list[Skill]) -> None:
    global GLOBAL_SKILLS
    GLOBAL_SKILLS = {s.name: s for s in skills}

@tool()
def lookup_skill(skill_name: str) -> tuple[str, list[AgentTool]]:
    """工具函数：查询技能的详细说明和使用指南，并返回该技能的工具.

    Args:
        skill_name: 技能名称

    Returns:
        元组 (技能的详细说明和使用指南, 该技能的工具列表)
    """
    return _do_lookup(skill_name)

def _do_lookup(skill_name: str) -> tuple[str, list[AgentTool]]:
    """内部函数：执行技能查询并返回说明和工具列表.
    
    Args: 
        skill_name: 技能名称
    Returns:
        技能说明和工具列表
    """
    skill = GLOBAL_SKILLS.get(skill_name)
    if not skill:
        return f"Error: Skill '{skill_name}' not found.", []
    return f"""
*** Skill Guide: {skill.name} ***
Description: {skill.description}

### Detailed Instructions:
{skill.instruction}

### Available Tools:
{', '.join([t.name for t in skill.tools])}
""", skill.tools
