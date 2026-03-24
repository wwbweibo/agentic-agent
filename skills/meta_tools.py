
from langchain_core.tools import tool

from .loader import Skill

# 全局存储已加载的 Skills，以便工具查找
# 在实际应用中，可以使用 ContextVar 或其它机制传递
GLOBAL_SKILLS: dict[str, Skill] = {}

def set_global_skills(skills): # noqa
    global GLOBAL_SKILLS
    for s in skills:
        GLOBAL_SKILLS[s.name] = s

@tool
def lookup_skill(skill_name: str) -> str:
    """使用该工具来获取技能具体说明.

    当你需要使用某个技能时，首先使用该工具来查询该技能的详细使用说明和能力范围，以确保你正确使用它。
    """
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
