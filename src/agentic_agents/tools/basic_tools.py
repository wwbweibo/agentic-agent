from datetime import datetime

from .base import create_tool

current_time_tool = create_tool(
    name="current_time",
    description="获取当前的日期和时间",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    func=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)
