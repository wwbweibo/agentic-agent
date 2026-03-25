from datetime import datetime

from .base import tool

@tool()
def current_time() -> str:
    """获取当前的日期和时间."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")