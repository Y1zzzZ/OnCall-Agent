"""
MCP 跨语言集成测试脚本 (Python Agent -> Java MCP Server)

这个脚本会启动我们的 Orchestrator，去连接你 8080 端口运行的 Java 服务器，
并尝试发起一次包含工具调用的对话。

运行前请确保：
1. Java MCP Server 已在 http://localhost:8080 启动
2. 已设置 DASHSCOPE_API_KEY 环境变量
"""

import asyncio
import os
from loguru import logger
from my_agent.services.search_service import OrchestratorService

async def test_java_mcp_link():
    # 1. 初始化中枢，配置你的 Java 服务器地址
    # 如果你本地 8080 启动了，我们就连 8080
    mcp_endpoint = "http://127.0.0.1:8085"
    logger.info(f"🚀 正在尝试连接 Mock MCP 服务器: {mcp_endpoint}")
    
    orchestrator = OrchestratorService(mcp_urls=[mcp_endpoint])
    
    # 诊断：用普通的 requests 试一下
    import requests
    try:
        diag = requests.post("http://127.0.0.1:8085/mcp/tools/list").json()
        logger.warning(f"DIAGNOSTIC REQUESTS: {diag}")
    except Exception as e:
        logger.error(f"DIAGNOSTIC FAILED: {e}")

    await orchestrator.initialize()
    
    # 2. 准备一个专门针对 Java 工具的问题
    # 问题 1: 测年假工具
    query1 = "帮我查一下用户 user_12345 的年假余额是多少？"
    logger.info(f"💬 提问: {query1}")
    response1 = await orchestrator.chat(query1)
    print(f"\n[AI 回答 - 年假]:\n{response1}\n")
    
    print("-" * 50)
    
    # 问题 2: 测订单工具
    query2 = "我的订单 ORD-6666 现在是什么状态？什么时候能送到？"
    logger.info(f"💬 提问: {query2}")
    response2 = await orchestrator.chat(query2)
    print(f"\n[AI 回答 - 订单]:\n{response2}\n")

if __name__ == "__main__":
    if "DASHSCOPE_API_KEY" not in os.environ:
        # 降级提醒，虽然代码里有 fallback，但显式设置更好
        os.environ["DASHSCOPE_API_KEY"] = "sk-f007c0ff34ab4b93882e5434ef25102c"
        
    asyncio.run(test_java_mcp_link())
