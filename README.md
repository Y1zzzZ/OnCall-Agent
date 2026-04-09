# OnCall Agent

> 企业级智能运维助手，支持 RAG 知识库问答和 AIOps 智能诊断

## ✨ 功能特性

### 智能对话
- 基于 LangChain 的多轮对话能力
- 流式输出，实时响应
- 支持 RAG 知识库检索增强

### RAG 知识库
- 向量检索增强问答
- 文档上传与自动索引
- 支持 Markdown、TXT 等格式文档

### AIOps 智能运维
- 基于 Plan-Execute-Replan 模式的自动故障诊断
- 根因分析
- 运维建议生成
- MCP 协议集成，支持日志查询、监控数据等工具

### Web 界面
- 现代化 UI 设计
- 多种对话模式：快速问答、流式对话
- 会话管理

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + LangChain + LangGraph |
| LLM | 阿里云 DashScope (通义千问) |
| 向量数据库 | Milvus |
| 工具协议 | MCP (Model Context Protocol) |

## 📡 API 接口

| 功能 | 方法 | 路径 |
|------|------|------|
| 普通对话 | POST | `/api/chat` |
| 流式对话 | POST | `/api/chat_stream` |
| AIOps 诊断 | POST | `/api/aiops` |
| 文件上传 | POST | `/api/upload` |
| 目录索�� | POST | `/api/index_directory` |
| 会话清除 | POST | `/api/chat/clear` |
| 健康检查 | GET | `/health` |

## 📁 项目结构

```
super_biz_agent_py/
├── app/                    # 应用核心
│   ├── main.py             # FastAPI 应用入口
│   ├── config.py           # 配置管理
│   ├── api/                # API 路由层
│   ├── services/           # 业务服务层（RAG、AIOps、向量处理）
│   ├── agent/             # Agent 模块（MCP 客户端、AIops 核心逻辑）
│   ├── models/             # 数据模型
│   ├── tools/              # Agent 工具集
│   ├── core/               # 核心组件（LLM 工厂、Milvus 客户端）
│   └── utils/              # 工具类
├── static/                 # Web 前端（纯静态）
├── mcp_servers/            # MCP 服务器（日志查询、监控数据）
├── aiops-docs/             # 运维知识库文档
├── .env.example            # 环境变量配置示例
├── docker-compose.yml       # Docker 服务配置
└── pyproject.toml          # 项目配置
```

## ⚙️ 配置说明

复制 `.env.example` 为 `.env`，填入你的配置：

```bash
# 阿里云 DashScope LLM 配置
DASHSCOPE_API_KEY=your-api-key
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-max

# Milvus 向量数据库
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断流程：

1. **Planner** - 制定诊断计划，生成诊断步骤
2. **Executor** - 执行诊断步骤，调用 MCP 工具
3. **Replanner** - 评估结果，动态调整后续步骤
4. **Report** - 生成诊断报告，包含根因分析和运维建议

## 📚 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph Plan-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [阿里云 DashScope](https://dashscope.aliyun.com/)
- [MCP 协议](https://modelcontextprotocol.io/)

## 📄 许可证

MIT License
