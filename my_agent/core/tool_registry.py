"""
工具挂载注册中心 (Tool Registry)
对应你架构图中的第二大核心拼图。

负责统一收集、鉴权、分发【内部 Python 工具】与【外部 MCP 工具】的大本营。
只要在这里注册了 Schema，大模型就能认识它。
"""

from typing import Dict, Any, List
from loguru import logger
from my_agent.core.mcp_client import HttpMCPClient

class ToolRegistry:
    def __init__(self):
        self.local_tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_client = HttpMCPClient()
        
    def register_local_tool(self, name: str, schema: dict, func: callable):
        """
        注册一个跑在本机内存里的纯 Python 工具（比如我们的核心 RAG 检索器）
        """
        self.local_tools[name] = {
            "schema": schema,
            "func": func 
        }
        logger.debug(f"[Registry] 本地神器已归位: {name}")
        
    async def initialize_mcp_servers(self, mcp_server_urls: List[str]):
        """
        系统启动时：遍历所有的远端 MCP Server，把全天下的工具都拉到手里
        """
        if not mcp_server_urls:
            return
            
        logger.info(f"[Registry] 准备扫荡 {len(mcp_server_urls)} 个外部 MCP 节点库: {mcp_server_urls}")
        for url in mcp_server_urls:
            try:
                tools_schema = await self.mcp_client.list_tools(url)
                logger.info(f"[Registry] 节点 [{url}] 返回了 {len(tools_schema)} 个工具。")
                for t_schema in tools_schema:
                    tool_name = t_schema.get("name")
                    if not tool_name: continue
                    
                    self.mcp_tools[tool_name] = {
                        "schema": t_schema, 
                        "server_url": url
                    }
                logger.success(f"[Registry] 成功从节点 [{url}] 收编 {len(tools_schema)} 个外包军团。")
            except Exception as e:
                logger.error(f"❌ [Registry] 从节点 [{url}] 抓取失败: {e}")
            
    def get_all_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        把【内家拳】和【外家帮】的所有兵器谱（Schema）混合在一起。
        严格打包成大模型（如 Qwen, OpenAI）要求的标准 function_call 入参格式。
        """
        standard_tools = []
        
        for name, info in self.local_tools.items():
            standard_tools.append({
                "type": "function",
                "function": info["schema"]
            })
            
        for name, info in self.mcp_tools.items():
            # 【核心修复】MCP 协议使用 inputSchema，但 OpenAI/DashScope 要求 parameters
            # 必须进行字段映射转换，否则大模型会直接忽略该工具
            mcp_schema = info["schema"]
            standard_tools.append({
                "type": "function",
                "function": {
                    "name": mcp_schema.get("name"),
                    "description": mcp_schema.get("description"),
                    "parameters": mcp_schema.get("inputSchema", {})
                }
            })
            
        logger.info(f"[Registry] 当前总兵力: 本地({len(self.local_tools)}) + 外部({len(self.mcp_tools)}) = {len(standard_tools)}")
        return standard_tools
        
    def is_local(self, tool_name: str) -> bool:
        return tool_name in self.local_tools
        
    def is_mcp(self, tool_name: str) -> bool:
        return tool_name in self.mcp_tools
        
    def get_local_function(self, tool_name: str) -> callable:
        """提取本地工具的真实执行指针"""
        return self.local_tools[tool_name]["func"]
        
    def get_mcp_server_url(self, tool_name: str) -> str:
        """获取负责执行该工具的外部结界 (URL)"""
        return self.mcp_tools[tool_name]["server_url"]
