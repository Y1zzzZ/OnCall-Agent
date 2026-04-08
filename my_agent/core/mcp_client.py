"""
MCP 客户端核心 (MCP HTTP Client)
对应 Java 版的: com.nageoffer.ai.ragent.rag.core.mcp.client.HttpMCPClient

负责通过网络（HTTP/SSE 等）向远端的 MCP Server（如我们的 mcp-server 服务）发送标准的 tools/list 和 tools/call 指令。
这里我们推荐使用 Python 现代异步库 `httpx`，以极致榨干高并发网络 I/O 性能。
(如果没有安装，请执行: pip install httpx)
"""

import httpx
import requests
import asyncio
import json
from loguru import logger
from typing import List, Dict, Any

class HttpMCPClient:
    def __init__(self, timeout: int = 15):
        """
        Args:
            timeout: 外部网络请求超时时间。设置合理的超时也是【熔断降级】的核心手段。
        """
        self.timeout = timeout

    async def list_tools(self, server_url: str) -> List[Dict[str, Any]]:
        """
        向远端 MCP Server 索要它的能力清单 (JSON Schema 列表)
        对应 MCP 协议的 `tools/list`
        """
        endpoint = f"{server_url}/mcp/tools/list"
        logger.debug(f"[MCP Client] 正在向 {endpoint} 请求工具清单...")
        
        try:
            # 采用 requests (同步) 并用 asyncio 模拟异步发送，这对本地 Windows 网络环境更稳健
            def sync_post():
                return requests.post(endpoint, json={}, timeout=10.0)
            
            response = await asyncio.to_thread(sync_post)
            logger.success(f"[MCP Client] POST {endpoint} -> Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.success(f"[MCP Client] 收到原始 JSON (via requests): {data}")
                return data.get("tools", [])
            else:
                # 最后的倔强：尝试 GET
                def sync_get():
                    return requests.get(endpoint, timeout=10.0)
                response = await asyncio.to_thread(sync_get)
                if response.status_code == 200:
                    return response.json().get("tools", [])
                return []
        except Exception as e:
            logger.error(f"❌ [MCP Client] 网络通信崩溃: {e}")
            return []

    async def call_tool(self, server_url: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        核心转发：向远端 MCP Server 发起真正的函数执行请求
        对应 MCP 协议的 `tools/call`
        """
        endpoint = f"{server_url}/mcp/tools/call"
        payload = {
            "name": tool_name,
            "arguments": arguments
        }
        
        logger.info(f"[MCP Client] 🚀 正在向 {server_url} 发射工具执行指令: {tool_name}")
        try:
            def sync_call():
                return requests.post(endpoint, json=payload, timeout=20.0)
            
            response = await asyncio.to_thread(sync_call)
            response.raise_for_status()
            data = response.json()
            return json.dumps(data, ensure_ascii=False) # Ensure JSON is returned as string for LLM
        except Exception as e:
            logger.error(f"❌ 调用 MCP 工具 [{tool_name}] 失败: {str(e)}")
            # The original error handling logic is preserved for consistency with LLM interaction
            # 【容错核心】：绝对不能向上抛出 Exception 让系统宕机！
            # 我们用人类语言包装错误，原封不动返回给大模型。
            # 大模型看到报错文本后，它的机制会迫使它进行“反思 (Reflection)”并尝试修复参数重新调用！
            return f"Error: 远程工具调用失败，错误详情为 {str(e)}。大模型请检查参数是否符合要求或告知用户操作失败。"
