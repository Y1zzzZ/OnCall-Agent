"""
极简 Mock MCP Server (Streamable HTTP 方案)

这是一个基于 Python 的模拟服务器，用于演示你的 Python Agent 如何通过 HTTP 协议
调用远端（如 Java Spring Boot）提供的工具。

运行方式：
pip install fastapi uvicorn
python my_agent/core/mock_mcp_server.py
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="OnCallAgent Mock MCP Server")

# 1. 模拟工具列表 (对应 Java 版的 EnterpriseTools)
TOOLS = [
    {
        "name": "getUserAnnualLeave",
        "description": "Check user annual leave balance. Use this when user asks about leave or remaining days.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "userId": {"type": "string", "description": "The user ID, e.g. user_12345"}
            },
            "required": ["userId"]
        }
    },
    {
        "name": "getOrderStatus",
        "description": "Query order logistics status. Use this when user asks about order progress or delivery.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "orderId": {"type": "string", "description": "The order ID, e.g. ORD-12345"}
            },
            "required": ["orderId"]
        }
    }
]

@app.get("/mcp/tools/list")
@app.post("/mcp/tools/list")
@app.get("/tools")
@app.post("/tools")
async def list_tools():
    """MCP 协议：获取工具定义列表"""
    print("收到工具列表请求！")
    return {"tools": TOOLS}

@app.post("/mcp/tools/call")
@app.post("/tools/call")
async def call_tool(request: Request):
    """MCP 协议：执行对应的工具"""
    body = await request.json()
    name = body.get("name")
    arguments = body.get("arguments", {})

    print(f"接到指令！准备执行工具: {name} | 参数: {arguments}")

    if name == "getUserAnnualLeave":
        user_id = arguments.get("userId", "unknown")
        # 模拟 Java 版的逻辑
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"用户 {user_id} 的年假余额为 5 天（总天数 10 天，已使用 5 天）。"
                }
            ]
        }
    
    elif name == "getOrderStatus":
        order_id = arguments.get("orderId", "ORD-UNKNOWN")
        # 模拟业务中台数据
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"订单 {order_id} 目前状态为：【运输中】，当前位于北京市朝阳区分拨中心，预计明日送达。"
                }
            ]
        }

    return JSONResponse(status_code=404, content={"error": "Tool not found"})

if __name__ == "__main__":
    # 启动在 8080 端口，模拟远端微服务
    uvicorn.run(app, host="0.0.0.0", port=8085)
