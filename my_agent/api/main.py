"""
FastAPI 后端网关 (API Gateway)

对接 React 18 前端的 SSE 流式聊天与会话管理接口。
前端 SSE 聊天请求格式：GET /rag/v3/chat?question=...&conversationId=...
前端期望的 SSE 事件类型：meta, message, finish, done, cancel, title, error

启动方法：
    uvicorn my_agent.api.main:app --host 0.0.0.0 --port 9090 --reload
"""

import json
import uuid
import asyncio
import os
import tempfile
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from my_agent.services.search_service import OrchestratorService
from my_agent.ingestion.ingest_docs import IngestionPipeline as DocIngestionPipeline
from loguru import logger

# ──────────────────────────────────────────────────────────────
# 全局 Orchestrator 单例（服务启动时初始化一次）
# ──────────────────────────────────────────────────────────────
_orchestrator: Optional[OrchestratorService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan 事件：启动时初始化 Orchestrator（扫描 MCP 工具）"""
    global _orchestrator
    logger.info("🚀 FastAPI 后端启动，正在初始化 Orchestrator 中枢...")
    _orchestrator = OrchestratorService(mcp_urls=["http://localhost:8085"])
    await _orchestrator.initialize()
    logger.success("✅ Orchestrator 初始化完成，准备接收前端请求！")
    yield
    logger.info("👋 FastAPI 后端正在关闭...")


app = FastAPI(
    title="OnCallAgent API",
    description="企业级 RAG + MCP 智能对话后端",
    version="1.0.0",
    lifespan=lifespan
)

# ──────────────────────────────────────────────────────────────
# CORS：允许前端本地开发服务器（Vite 默认 5173）访问
# ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段放开，生产环境改成具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# 内存中的会话存储（轻量调试用）
# 生产环境可换成 Redis 或数据库
# ──────────────────────────────────────────────────────────────
# 格式: { conversationId: { "title": str, "messages": [...], "lastTime": str } }
_sessions: dict = {}
_message_store: dict = {}  # { messageId: { ...message_obj } }


def _now_iso() -> str:
    return datetime.now().isoformat()

def _get_or_create_session(conv_id: Optional[str]) -> str:
    """如果没有传 conversationId 就新建一个"""
    if conv_id and conv_id in _sessions:
        return conv_id
    new_id = conv_id or str(uuid.uuid4())
    _sessions[new_id] = {
        "conversationId": new_id,
        "title": "新对话",
        "messages": [],
        "lastTime": _now_iso()
    }
    return new_id


# ──────────────────────────────────────────────────────────────
# SSE 辅助函数
# ──────────────────────────────────────────────────────────────
def _sse_event(event: str, data: object) -> str:
    """格式化一条 SSE 消息"""
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


# ──────────────────────────────────────────────────────────────
# 系统配置接口
# GET /rag/settings
# ──────────────────────────────────────────────────────────────
@app.get("/api/ragent/rag/settings")
async def get_rag_settings():
    """
    提供系统配置信息，包括可选模型列表。
    对齐前端 settingsService.ts 中的 SystemSettings 接口。
    """
    return {
        "rag": {
            "default": {
                "collectionName": "mydoc_knowledge_base",
                "dimension": 1024,
                "metricType": "COSINE"
            },
            "queryRewrite": {"enabled": True, "maxHistoryMessages": 10, "maxHistoryChars": 1000},
            "memory": {
                "historyKeepTurns": 10,
                "summaryStartTurns": 5,
                "summaryEnabled": True,
                "ttlMinutes": 60,
                "summaryMaxChars": 1000,
                "titleMaxLength": 20
            }
        },
        "ai": {
            "embedding": {
                "defaultModel": "text-embedding-v3",
                "candidates": [
                    {
                        "id": "text-embedding-v3",
                        "provider": "DashScope",
                        "model": "text-embedding-v3",
                        "enabled": True,
                        "dimension": 1024
                    }
                ]
            },
            "chat": {
                "defaultModel": "qwen-plus",
                "candidates": [
                    {"id": "qwen-plus", "provider": "DashScope", "model": "qwen-plus", "enabled": True},
                    {"id": "qwen-turbo", "provider": "DashScope", "model": "qwen-turbo", "enabled": True}
                ]
            },
            "rerank": {
                "defaultModel": "gte-reranker-v2-m3",
                "candidates": [
                    {"id": "gte-reranker-v2-m3", "provider": "DashScope", "model": "gte-reranker-v2-m3", "enabled": True}
                ]
            }
        }
    }


# ──────────────────────────────────────────────────────────────
# 核心聊天接口（SSE 流式）
# GET /rag/v3/chat?question=...&conversationId=...
# ──────────────────────────────────────────────────────────────
@app.get("/api/ragent/rag/v3/chat")
async def chat_stream(
    question: str = Query(..., description="用户问题"),
    conversationId: Optional[str] = Query(None, description="会话 ID，不传则新建"),
    deepThinking: Optional[bool] = Query(None, description="是否开启深度思考（暂未使用）"),
    kbId: Optional[str] = Query(None, description="所属知识库 ID")
):
    """
    前端通过 GET SSE 请求此接口。
    事件流顺序：meta -> message(×N) -> finish -> done
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator 尚未初始化，请稍后再试")

    # 解析目标集合
    collection_name = None
    if kbId:
        kb_info = _knowledge_bases.get(kbId)
        if kb_info:
            collection_name = kb_info.get("collectionName")
            logger.info(f"🎯 聊天上下文定向至知识库: {kb_info['name']} ({collection_name})")
    
    if not collection_name:
        # 默认回退到核心知识库
        collection_name = _knowledge_bases.get(_DEFAULT_KB_ID, {}).get("collectionName")

    conv_id = _get_or_create_session(conversationId)
    task_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    async def event_generator():
        # 1. 发送 meta 事件（告知前端本次的 conversationId 和 taskId）
        yield _sse_event("meta", {"conversationId": conv_id, "taskId": task_id})

        # 2. 流式调用后端 Orchestrator，每个 token 实时推送
        try:
            full_answer = ""
            async for token in _orchestrator.chat_stream(
                question,
                session_id=conv_id,
                collection_name=collection_name
            ):
                full_answer += token
                yield _sse_event("message", {"type": "response", "delta": token})
                await asyncio.sleep(0)
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})
            yield _sse_event("done", "")
            return

        # 3. 持久化到内存
        title = question[:20] + ("..." if len(question) > 20 else "")
        _sessions[conv_id]["lastTime"] = _now_iso()
        if _sessions[conv_id]["title"] == "新对话":
            _sessions[conv_id]["title"] = title

        user_msg_id = str(uuid.uuid4())
        bot_msg_id = message_id

        user_msg = {
            "id": user_msg_id,
            "conversationId": conv_id,
            "role": "user",
            "content": question,
            "vote": None,
            "createTime": _now_iso()
        }
        bot_msg = {
            "id": bot_msg_id,
            "conversationId": conv_id,
            "role": "assistant",
            "content": full_answer,
            "vote": None,
            "createTime": _now_iso()
        }
        _sessions[conv_id]["messages"].extend([user_msg, bot_msg])
        _message_store[user_msg_id] = user_msg
        _message_store[bot_msg_id] = bot_msg

        # 4. 发送 finish 事件（包含 messageId 和 title，前端会据此更新侧边栏标题）
        yield _sse_event("finish", {"messageId": bot_msg_id, "title": title})

        # 5. 发送 done 事件（告知前端流结束）
        yield _sse_event("done", "")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲，确保 SSE 即时推送
        }
    )


# ──────────────────────────────────────────────────────────────
# 会话管理接口
# ──────────────────────────────────────────────────────────────
@app.get("/api/ragent/conversations")
async def list_conversations():
    """返回所有会话列表"""
    result = [
        {
            "conversationId": sid,
            "title": s["title"],
            "lastTime": s["lastTime"]
        }
        for sid, s in sorted(_sessions.items(), key=lambda x: x[1]["lastTime"], reverse=True)
    ]
    return {"code": "0", "data": result}


@app.delete("/api/ragent/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除指定会话"""
    if conv_id in _sessions:
        del _sessions[conv_id]
    return {"code": "0", "data": None}


class RenameBody(BaseModel):
    title: str

@app.put("/api/ragent/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: RenameBody):
    """重命名会话"""
    if conv_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    _sessions[conv_id]["title"] = body.title
    return {"code": "0", "data": None}


@app.get("/api/ragent/conversations/{conv_id}/messages")
async def list_messages(conv_id: str):
    """返回指定会话的历史消息"""
    if conv_id not in _sessions:
        return {"code": "0", "data": []}
    messages = _sessions[conv_id].get("messages", [])
    return {"code": "0", "data": messages}


# ──────────────────────────────────────────────────────────────
# 消息反馈接口（点赞/踩）
# ──────────────────────────────────────────────────────────────
class FeedbackBody(BaseModel):
    vote: int  # 1=like, -1=dislike

@app.post("/api/ragent/conversations/messages/{message_id}/feedback")
async def submit_feedback(message_id: str, body: FeedbackBody):
    """更新指定消息的反馈"""
    if message_id in _message_store:
        _message_store[message_id]["vote"] = body.vote
    return {"code": "0", "data": None}


# ──────────────────────────────────────────────────────────────
# 停止任务接口（前端发送停止指令时调用）
# ──────────────────────────────────────────────────────────────
@app.post("/api/ragent/rag/v3/stop")
async def stop_task(taskId: str = Query(...)):
    """前端取消生成时调用，这里做简单记录（实际流已在前端断开连接时中断）"""
    return {"code": "0", "data": None}


# ──────────────────────────────────────────────────────────────
# 认证接口（简化版，绕过真实认证）
# ──────────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    username: str
    password: str

@app.post("/api/ragent/auth/login")
async def login(body: LoginBody):
    """
    开发阶段：任何用户名+密码都直接登录成功。
    生产阶段可接入 LDAP / JWT 等认证体系。
    """
    return {
        "code": "0",
        "data": {
            "id": "dev-user-001",
            "username": body.username,
            "role": "admin",
            "token": f"dev-token-{body.username}"
        }
    }

@app.post("/api/ragent/auth/logout")
async def logout():
    return {"code": "0", "data": None}

@app.get("/api/ragent/user/me")
async def get_current_user(request: Request):
    """返回当前登录用户信息（从 Authorization header 中解析）"""
    token = request.headers.get("Authorization", "")
    username = token.replace("dev-token-", "") if token.startswith("dev-token-") else "admin"
    return {
        "code": "0",
        "data": {
            "id": "dev-user-001",
            "username": username,
            "role": "admin"
        }
    }



# ──────────────────────────────────────────────────────────────
# 文档入库 (Ingestion) 接口
# 前端 admin/ingestion 页面调用
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 知识库管理 (Knowledge Base) 接口
# 对接 frontend/src/services/knowledgeService.ts
# ──────────────────────────────────────────────────────────────

# 内存存储（调试用）
_knowledge_bases: dict = {}
_kb_documents: dict = {} # { kbId: [ {doc} ] }

# 预置一个默认知识库
_DEFAULT_KB_ID = "default-kb"
_knowledge_bases[_DEFAULT_KB_ID] = {
    "id": _DEFAULT_KB_ID,
    "name": "核心业务知识库",
    "embeddingModel": "text-embedding-v3",
    "collectionName": "mydoc_knowledge_base",
    "documentCount": 0,
    "createTime": _now_iso(),
    "updateTime": _now_iso()
}

@app.get("/api/ragent/knowledge-base")
async def list_knowledge_bases(current: int = 1, size: int = 10, name: str = ""):
    items = list(_knowledge_bases.values())
    if name:
        items = [kb for kb in items if name in kb["name"]]
    
    total = len(items)
    start = (current - 1) * size
    page_items = items[start: start + size]
    
    return {"code": "0", "data": {
        "records": page_items,
        "total": total,
        "size": size,
        "current": current,
        "pages": max(1, (total + size - 1) // size)
    }}

@app.get("/api/ragent/knowledge-base/{kb_id}")
async def get_knowledge_base(kb_id: str):
    kb = _knowledge_bases.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return {"code": "0", "data": kb}

@app.post("/api/ragent/knowledge-base")
async def create_knowledge_base(body: dict):
    kb_id = str(uuid.uuid4())
    name = body.get("name", "未命名知识库")
    kb = {
        "id": kb_id,
        "name": name,
        "embeddingModel": body.get("embeddingModel", "text-embedding-v3"),
        "collectionName": f"kb_{kb_id.replace('-', '_')}",
        "documentCount": 0,
        "createTime": _now_iso(),
        "updateTime": _now_iso()
    }
    _knowledge_bases[kb_id] = kb
    # 初始化 Milvus 集合
    try:
        from my_agent.core.milvus_manager import MilvusManager
        mm = MilvusManager()
        mm.init_collection(kb["collectionName"])
    except Exception as e:
        from loguru import logger
        logger.error(f"初始化 Milvus 集合失败: {e}")

    return {"code": "0", "data": kb_id}

@app.put("/api/ragent/knowledge-base/{kb_id}")
async def update_knowledge_base(kb_id: str, body: dict):
    if kb_id not in _knowledge_bases:
        raise HTTPException(status_code=404, detail="知识库不存在")
    _knowledge_bases[kb_id].update({
        "name": body.get("name", _knowledge_bases[kb_id]["name"]),
        "embeddingModel": body.get("embeddingModel", _knowledge_bases[kb_id].get("embeddingModel")),
        "updateTime": _now_iso()
    })
    return {"code": "0", "data": None}

@app.delete("/api/ragent/knowledge-base/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    _knowledge_bases.pop(kb_id, None)
    _kb_documents.pop(kb_id, None)
    return {"code": "0", "data": None}

# --- 文档子资源 ---

@app.get("/api/ragent/knowledge-base/{kb_id}/docs")
async def list_kb_documents(kb_id: str, pageNo: int = 1, pageSize: int = 10, keyword: str = ""):
    docs = _kb_documents.get(kb_id, [])
    if keyword:
        docs = [d for d in docs if keyword in d["docName"]]
    
    total = len(docs)
    start = (pageNo - 1) * pageSize
    page_items = docs[start: start + pageSize]
    
    return {"code": "0", "data": {
        "records": page_items,
        "total": total,
        "size": pageSize,
        "current": pageNo,
        "pages": max(1, (total + pageSize - 1) // pageSize)
    }}

@app.post("/api/ragent/knowledge-base/{kb_id}/docs/upload")
async def upload_kb_document(
    kb_id: str,
    file: UploadFile = File(...),
    sourceType: str = Form("file"),
    processMode: str = Form("pipeline")
):
    """
    对接前端 uploadDocument 调用的接口。
    """
    from loguru import logger
    kb = _knowledge_bases.get(kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    doc_id = str(uuid.uuid4())
    filename = file.filename or "upload.txt"
    
    # 记录文档元数据
    new_doc = {
        "id": doc_id,
        "kbId": kb_id,
        "docName": filename,
        "status": "PROCESSING",
        "createTime": _now_iso(),
        "updateTime": _now_iso()
    }
    if kb_id not in _kb_documents:
        _kb_documents[kb_id] = []
    _kb_documents[kb_id].append(new_doc)

    # 异步开始入库
    tmp_dir = tempfile.mkdtemp(prefix=f"kb_{kb_id}_")
    tmp_path = os.path.join(tmp_dir, filename)
    
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        collection_name = kb["collectionName"]

        def _run_ingestion():
            pipeline = DocIngestionPipeline()
            pipeline.run_pipeline(tmp_dir, collection_name=collection_name)

        await asyncio.to_thread(_run_ingestion)

        # 更新状态
        new_doc.update({
            "status": "COMPLETED",
            "chunkCount": max(1, len(content) // 500),
            "updateTime": _now_iso()
        })
        kb["documentCount"] += 1
        logger.success(f"📄 文档 [{filename}] 已成功入库知识库 [{kb['name']}]")

    except Exception as e:
        new_doc.update({
            "status": "FAILED",
            "errorMessage": str(e),
            "updateTime": _now_iso()
        })
        logger.error(f"❌ 文档入库失败: {e}")
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except: pass

    return {"code": "0", "data": new_doc}


# ──────────────────────────────────────────────────────────────
# 文档子资源接口（stub，保证主流程不报错）
# 前端文档详情/分块管理调用
# ──────────────────────────────────────────────────────────────

@app.get("/api/ragent/knowledge-base/docs/{doc_id}")
async def get_document(doc_id: str):
    for docs in _kb_documents.values():
        for doc in docs:
            if doc["id"] == doc_id:
                return {"code": "0", "data": doc}
    raise HTTPException(status_code=404, detail="文档不存在")


@app.put("/api/ragent/knowledge-base/docs/{doc_id}")
async def update_document(doc_id: str, body: dict):
    for docs in _kb_documents.values():
        for doc in docs:
            if doc["id"] == doc_id:
                if "docName" in body:
                    doc["docName"] = body["docName"]
                doc["updateTime"] = _now_iso()
                return {"code": "0", "data": None}
    raise HTTPException(status_code=404, detail="文档不存在")


@app.delete("/api/ragent/knowledge-base/docs/{doc_id}")
async def delete_document(doc_id: str):
    for kb_id, docs in list(_kb_documents.items()):
        for i, doc in enumerate(docs):
            if doc["id"] == doc_id:
                docs.pop(i)
                kb = _knowledge_bases.get(kb_id)
                if kb and kb["documentCount"] > 0:
                    kb["documentCount"] -= 1
                return {"code": "0", "data": None}
    raise HTTPException(status_code=404, detail="文档不存在")


@app.post("/api/ragent/knowledge-base/docs/{doc_id}/chunk")
async def start_document_chunk(doc_id: str):
    return {"code": "0", "data": None}


@app.patch("/api/ragent/knowledge-base/docs/{doc_id}/enable")
async def enable_document(doc_id: str, value: bool = Query(False)):
    return {"code": "0", "data": None}


@app.post("/api/ragent/knowledge-base/docs/{doc_id}/chunks/rebuild")
async def rebuild_chunks(doc_id: str):
    return {"code": "0", "data": None}


@app.get("/api/ragent/knowledge-base/docs/{doc_id}/chunks/logs")
async def get_chunk_logs(doc_id: str, current: int = 1, size: int = 10):
    return {"code": "0", "data": {
        "records": [], "total": 0, "size": size, "current": current, "pages": 1
    }}


# ──────────────────────────────────────────────────────────────
# 文档分块 CRUD（stub）
# ──────────────────────────────────────────────────────────────

@app.get("/api/ragent/knowledge-base/docs/{doc_id}/chunks")
async def list_chunks(doc_id: str, current: int = 1, size: int = 10, enabled: bool = True):
    return {"code": "0", "data": {
        "records": [], "total": 0, "size": size, "current": current, "pages": 1
    }}


@app.post("/api/ragent/knowledge-base/docs/{doc_id}/chunks")
async def create_chunk(doc_id: str, body: dict):
    return {"code": "0", "data": {
        "id": str(uuid.uuid4()), "docId": doc_id, "content": body.get("content", ""),
        "chunkIndex": body.get("index", 0), "createTime": _now_iso(), "updateTime": _now_iso()
    }}


@app.put("/api/ragent/knowledge-base/docs/{doc_id}/chunks/{chunk_id}")
async def update_chunk(doc_id: str, chunk_id: str, body: dict):
    return {"code": "0", "data": None}


@app.delete("/api/ragent/knowledge-base/docs/{doc_id}/chunks/{chunk_id}")
async def delete_chunk(doc_id: str, chunk_id: str):
    return {"code": "0", "data": None}


@app.patch("/api/ragent/knowledge-base/docs/{doc_id}/chunks/{chunk_id}/enable")
async def enable_chunk(doc_id: str, chunk_id: str, value: bool = Query(False)):
    return {"code": "0", "data": None}


@app.patch("/api/ragent/knowledge-base/docs/{doc_id}/chunks/{chunk_id}/disable")
async def disable_chunk(doc_id: str, chunk_id: str):
    return {"code": "0", "data": None}


@app.patch("/api/ragent/knowledge-base/docs/{doc_id}/chunks/batch/enable")
async def batch_enable_chunks(doc_id: str, chunkIds: str = Query(None)):
    return {"code": "0", "data": None}


@app.patch("/api/ragent/knowledge-base/docs/{doc_id}/chunks/batch/disable")
async def batch_disable_chunks(doc_id: str, chunkIds: str = Query(None)):
    return {"code": "0", "data": None}


# ──────────────────────────────────────────────────────────────
# Ingestion Pipeline 接口（stub，保证上传弹窗不报错）
# ──────────────────────────────────────────────────────────────

@app.get("/api/ragent/ingestion/pipelines")
async def list_pipelines(pageNo: int = 1, pageSize: int = 100, keyword: str = ""):
    return {"code": "0", "data": {
        "records": [], "total": 0, "size": pageSize, "current": pageNo, "pages": 1
    }}


@app.get("/api/ragent/ingestion/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    raise HTTPException(status_code=404, detail="Pipeline不存在")


@app.post("/api/ragent/ingestion/pipelines")
async def create_pipeline(body: dict):
    return {"code": "0", "data": str(uuid.uuid4())}


@app.put("/api/ragent/ingestion/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str, body: dict):
    return {"code": "0", "data": None}


@app.delete("/api/ragent/ingestion/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    return {"code": "0", "data": None}


@app.get("/api/ragent/ingestion/tasks")
async def list_tasks(pageNo: int = 1, pageSize: int = 10, status: str = ""):
    return {"code": "0", "data": {
        "records": [], "total": 0, "size": pageSize, "current": pageNo, "pages": 1
    }}


@app.get("/api/ragent/ingestion/tasks/{task_id}")
async def get_task(task_id: str):
    raise HTTPException(status_code=404, detail="任务不存在")


@app.get("/api/ragent/ingestion/tasks/{task_id}/nodes")
async def get_task_nodes(task_id: str):
    return {"code": "0", "data": []}


# ──────────────────────────────────────────────────────────────
# 健康检查
# ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "orchestrator": "initialized" if _orchestrator else "pending"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("my_agent.api.main:app", host="0.0.0.0", port=9090, reload=True)
