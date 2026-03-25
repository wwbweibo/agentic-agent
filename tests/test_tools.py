"""tools 模块单元测试 - AgentTool、create_tool、@tool 装饰器."""

import asyncio

import pytest

from agentic_agents.tools.base import AgentTool, create_tool, tool


class TestAgentTool:
    """AgentTool 数据类测试."""

    def test_to_dict(self):
        t = AgentTool(
            name="my_tool",
            description="A test tool",
            func=lambda: None,
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        d = t.to_dict()
        assert d == {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        }

    def test_to_dict_default_parameters(self):
        t = AgentTool(name="t", description="d", func=lambda: None)
        d = t.to_dict()
        assert d["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_execute_sync(self):
        def add(a: int, b: int) -> int:
            return a + b

        t = AgentTool(name="add", description="add", func=add)
        result = t.execute(a=1, b=2)
        assert result == 3

    def test_execute_async(self):
        async def greet(name: str) -> str:
            return f"hello {name}"

        t = AgentTool(name="greet", description="greet", func=greet)
        coro = t.execute(name="world")
        assert asyncio.iscoroutine(coro)
        result = asyncio.get_event_loop().run_until_complete(coro)
        assert result == "hello world"

    def test_tags_default_empty(self):
        t = AgentTool(name="t", description="d", func=lambda: None)
        assert t.tags == []


class TestCreateTool:
    """create_tool 工厂函数测试."""

    def test_creates_agent_tool(self):
        t = create_tool(
            name="echo",
            description="echo input",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
            func=lambda msg: msg,
        )
        assert isinstance(t, AgentTool)
        assert t.name == "echo"
        assert t.tags == []

    def test_with_tags(self):
        t = create_tool(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {}},
            func=lambda: None,
            tags=["test"],
        )
        assert t.tags == ["test"]


class TestToolDecorator:
    """@tool 装饰器测试."""

    def test_basic_decorator(self):
        @tool()
        def my_func(x: str) -> str:
            """My description."""
            return x

        assert isinstance(my_func, AgentTool)
        assert my_func.name == "my_func"
        assert my_func.description == "My description."

    def test_custom_name_and_description(self):
        @tool(name="custom_name", description="Custom desc")
        def func():
            pass

        assert func.name == "custom_name"
        assert func.description == "Custom desc"

    def test_auto_parameters_from_signature(self):
        @tool()
        def search(query: str, count: int = 5) -> str:
            """Search."""
            return query

        params = search.parameters
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "count" in params["properties"]
        assert params["properties"]["count"]["type"] == "integer"
        # query 无默认值应为 required，count 有默认值不应为 required
        assert "query" in params["required"]
        assert "count" not in params["required"]

    def test_auto_parameters_type_mapping(self):
        @tool()
        def func(a: str, b: int, c: float, d: bool):
            pass

        props = func.parameters["properties"]
        assert props["a"]["type"] == "string"
        assert props["b"]["type"] == "integer"
        assert props["c"]["type"] == "number"
        assert props["d"]["type"] == "boolean"

    def test_complex_type_defaults_to_string(self):
        @tool()
        def func(data: list):
            pass

        assert func.parameters["properties"]["data"]["type"] == "string"

    def test_no_annotation_defaults_to_string(self):
        @tool()
        def func(x):
            pass

        assert func.parameters["properties"]["x"]["type"] == "string"
        assert "x" in func.parameters["required"]

    def test_explicit_parameters_override_auto(self):
        custom_params = {
            "type": "object",
            "properties": {"custom": {"type": "integer"}},
            "required": ["custom"],
        }

        @tool(parameters=custom_params)
        def func(x: str):
            pass

        assert func.parameters == custom_params

    def test_decorator_with_tags(self):
        @tool(tags=["search", "web"])
        def func():
            pass

        assert func.tags == ["search", "web"]


class TestBasicTools:
    """basic_tools 模块测试."""

    def test_current_time_returns_string(self):
        from agentic_agents.tools.basic_tools import current_time

        assert isinstance(current_time, AgentTool)
        result = current_time.func()
        assert isinstance(result, str)
        # 格式: YYYY-MM-DD HH:MM:SS
        assert len(result) == 19
