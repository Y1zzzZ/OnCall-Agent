"""
文档加载模块

负责根据不同文件类型，读取并解析原始文档。
"""

from .loader_router import route_and_load

__all__ = ["route_and_load"]
