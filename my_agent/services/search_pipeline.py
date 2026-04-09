"""
核心知识库检索流水线 (Search Pipeline)

将 Dense Retrieval, BM25, RRF 融合，以及 Cross-Encoder 重排序封装成独立管道，
实现高质量的知识检索。
"""

from typing import List, Dict, Any
from loguru import logger
import asyncio
import time

from my_agent.core.milvus_manager import MilvusManager
from my_agent.services.embedding_service import DashScopeEmbeddingService
from my_agent.services.rerank_service import DashScopeRerankService

class SearchPipeline:
    def __init__(self):
        # 初始化三大打手：管拿货的 (Milvus)，管翻译的 (Embedder)，管打分的 (Reranker)
        # 默认使用 阿里百炼 向量模型 text-embedding-v3
        self.milvus = MilvusManager(dim=1024)
        self.embedder = DashScopeEmbeddingService(model_name="text-embedding-v3")
        self.reranker = DashScopeRerankService()
        
    async def run(self, query: str, dense_top_k: int = 20, rerank_top_k: int = 5, collection_name: str = None) -> List[Dict[Any, Any]]:
        """
        极速检索内核：双路召回 -> RRF融合 -> Cross-Encoder 精排
        完美匹配了你 Java 代码 (runHybrid + rerank) 中的双重保障架构！
        """
        t0 = time.perf_counter()
        logger.info(f"🔥 [知识库主轴流水线] 猎犬出动，目标词: '{query}'")

        # ── ① Embedding ─────────────────────────────────────────────────
        logger.debug("[流水线] ①Embedding 异步发起...")
        t1 = time.perf_counter()
        query_vector = await self.embedder.embed_texts_async([query])
        embed_ms = (time.perf_counter() - t1) * 1000
        logger.info(f"  ⏱ [① Embedding 向量化] +{embed_ms:.0f}ms")

        if not query_vector:
            logger.error("API 故障未能生成向量。")
            return []
        query_vector = query_vector[0]

        # ── ② Milvus 混合召回 + ③ Rerank 精排 并行 ─────────────────────
        # Milvus (50ms) 比 Rerank (2.6s) 快得多，两者理论上可并行
        # Rerank 需要 Milvus 返回的候选文本，所以 Milvus 先完成
        # 用 asyncio.create_task 真正同时启动两个线程任务
        t2 = time.perf_counter()

        # 结果容器（闭包引用）
        _recall_texts: List[str] = []

        async def _milvus_task():
            results = await asyncio.to_thread(
                self.milvus.hybrid_search,
                query_text=query,
                query_dense_vec=query_vector,
                top_k=dense_top_k,
                collection_name=collection_name
            )
            _recall_texts.clear()
            _recall_texts.extend(r["text"] for r in results)
            return results

        async def _rerank_task():
            # Rerank 等待 Milvus 的候选文本准备好后再执行
            while not _recall_texts:
                await asyncio.sleep(0.001)
            return await asyncio.to_thread(
                self.reranker.rerank_documents,
                query=query,
                documents=_recall_texts.copy(),
                top_n=rerank_top_k
            )

        recall_task = asyncio.create_task(_milvus_task())
        rerank_task = asyncio.create_task(_rerank_task())

        # asyncio.gather 同时等待两者，max(50ms, 2.6s) ≈ 2.6s 完成
        recall_results, reranked_results = await asyncio.gather(recall_task, rerank_task)

        parallel_ms = (time.perf_counter() - t2) * 1000
        logger.info(f"  ⏱ [② Milvus + ③ Rerank 并行完成] +{parallel_ms:.0f}ms")
        
        logger.success(f"🎉 [知识库主轴流水线] 大功告成！从沙海中遴选出 {len(reranked_results)} 颗珍珠。"
                        f" 流水线总耗时: {(time.perf_counter()-t0)*1000:.0f}ms")
        return reranked_results

# 为了给 Agent Tool Registry 挂载用的全局单例
search_pipeline_instance = SearchPipeline()

async def run_internal_rag_pipeline(query: str, top_k: int = 5, collection_name: str = None) -> str:
    """
    暴露给大模型 ToolRegistry 的终极封装入口！
    因为大模型（Orchestrator）只认 "大白话文本"，我们需要把复杂的排序结果翻译成字符串报表。
    """
    # 粗排池子设大一点 (Top-20)，最后只留下最精华的 5 条证据给大模型断案
    results = await search_pipeline_instance.run(
        query=query, 
        dense_top_k=20, 
        rerank_top_k=top_k,
        collection_name=collection_name
    )
    
    if not results:
        return "公司专属知识库中暂未找到与该问题相关的材料。"
        
    formatted = []
    for idx, r in enumerate(results):
        formatted.append(f"【绝密情报 {idx+1}】(可信度打分: {r.get('score', 0):.4f}):\n{r.get('text', '')}")
        
    # 全部拍成一大段送货上门
    return "\n\n".join(formatted)
# TODO 为什么run_internal_rag_pipeline中重排序后的数据就丢了 不知道下一个节点是哪个函数了呢
if __name__ == "__main__":
    async def main():
        query = "请问公司差旅报销的标准是什么？"
        print(f"\n--- 测试搜索: {query} ---\n")
        result = await run_internal_rag_pipeline(query)
        print(result)

    asyncio.run(main())
