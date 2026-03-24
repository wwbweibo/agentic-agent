import importlib.util
import os
import sys

import frontmatter
from langchain_core.tools import BaseTool


class Skill:
    """表示一个加载的技能 (Skill)."""
    def __init__(self, name: str, description: str, instruction: str, tools: list[BaseTool], path: str):
        self.name = name
        self.description = description
        self.instruction = instruction
        self.tools = tools
        self.path = path

    def __repr__(self):
        return f"<Skill name='{self.name}'>"

def load_skills_from_directory(skills_dir: str) -> list[Skill]:
    """从指定目录加载所有技能.

    每个子目录被视为一个技能，必须包含 skill.md 和 tools.py。
    """
    loaded_skills = []

    # 确保目录存在
    if not os.path.exists(skills_dir):
        print(f"Warning: Skills directory '{skills_dir}' does not exist.")
        return []

    # 遍历子目录
    for item in sorted(os.listdir(skills_dir)): # 使用 sorted 保证加载顺序一致
        skill_path = os.path.join(skills_dir, item)
        if not os.path.isdir(skill_path):
            continue

        # 检查必要文件
        md_path = os.path.join(skill_path, "skill.md")
        tools_path = os.path.join(skill_path, "tools.py")

        if not (os.path.exists(md_path) and os.path.exists(tools_path)):
            continue

        try:
            # 1. 解析 skill.md
            with open(md_path, encoding="utf-8") as f:
                post = frontmatter.load(f)
                name = str(post.metadata.get("name", item))
                description = str(post.metadata.get("description", ""))
                instruction = str(post.content) # Markdown 正文作为 instruction

            # 2. 动态加载 tools.py
            # 加载前先清理旧模块以支持热重载（可选，但在开发中很有用）
            module_name = f"skills.dynamic.{item}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, tools_path)
            module = importlib.util.module_from_spec(spec) # type: ignore
            sys.modules[module_name] = module # 注册到 sys.modules
            spec.loader.exec_module(module)

            # 3. 提取所有被 @tool 装饰的函数
            skill_tools = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # 兼容 LangChain 的 BaseTool 和 StructuredTool
                if isinstance(attr, BaseTool):
                    _tool: BaseTool = attr
                    _tool.tags = ['Provided by skill: ' + name]  # type: ignore
                    skill_tools.append(_tool)

            if not skill_tools:
                print(f"Warning: No tools found in {item}/tools.py")

            # 4. 创建 Skill 对象
            skill = Skill(name, description, instruction, skill_tools, skill_path)
            loaded_skills.append(skill)
            # print(f"Loaded skill: {name}")

        except Exception as e:
            print(f"Error loading skill '{item}': {e}")

    return loaded_skills
