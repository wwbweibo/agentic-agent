
from langchain_core.tools import tool


@tool
def current_time() -> str:
    """获取当前的日期和时间."""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
