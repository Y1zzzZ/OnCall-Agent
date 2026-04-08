"""多通道检索引擎 - 并行多路检索 + 后处理流水线

架构:
    用户查询
       │
       ▼
    MultiChannelRetrievalEngine
       │
       ├──→ VectorChannel       （并行 asyncio.gather）复用现有的 Milvus 向量检索
       ├──→ IntentChannel        关键词识别意图 + 向量检索
       └──→ KeywordChannel       BM25 关键词过滤 + 向量检索
       │
       ▼
    PostProcessorPipeline
       │
       ├──→ DeduplicationProcessor  (Order=1) doc_id + 内容64字 MD5 去重
       └──→ RerankProcessor         (Order=10) 通道权重加权 + 归一化重排
       │
       ▼
    最终 Top-K 结果
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.intent_resolver import RoutingTarget
from app.services.vector_embedding_service import vector_embedding_service


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """统一检索结果格式"""
    doc_id: str
    content: str
    score: float
    channel: str
    metadata: dict = field(default_factory=dict)

    def to_document(self) -> Document:
        """转换为 LangChain Document"""
        return Document(
            page_content=self.content,
            metadata={**self.metadata, "doc_id": self.doc_id, "channel": self.channel}
        )


# ---------------------------------------------------------------------------
# 检索通道抽象
# ---------------------------------------------------------------------------

class RetrievalChannel(ABC):
    """检索通道基类"""

    name: str = "base"

    @abstractmethod
    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """执行检索，返回该通道的候选结果"""
        pass


# ---------------------------------------------------------------------------
# 通道1: 向量全局检索（Milvus）
# ---------------------------------------------------------------------------

class VectorChannel(RetrievalChannel):
    """向量语义检索通道 — 基于 Milvus ANN 检索"""

    name = "vector"

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            query_vector = vector_embedding_service.embed_query(query)

            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

            retrieval_results = []
            for hits in results:
                for hit in hits:
                    retrieval_results.append(RetrievalResult(
                        doc_id=hit.entity.get("id"),
                        content=hit.entity.get("content", ""),
                        score=float(-hit.distance),  # L2距离转相似度
                        channel=self.name,
                        metadata=hit.entity.get("metadata", {}),
                    ))

            logger.debug(f"[VectorChannel] 召回 {len(retrieval_results)} 条")
            return retrieval_results

        except Exception as e:
            logger.error(f"[VectorChannel] 检索失败: {e}")
            return []


# ---------------------------------------------------------------------------
# 通道2: 意图定向检索（可插拔，支持新旧两套实现）
# ---------------------------------------------------------------------------

# 旧版轻量关键词意图映射（intent_enabled=True 且 intent_llm_enabled=False 时使用）
INTENT_KEYWORDS = {
    "operation": ["部署", "启动", "停止", "重启", "扩容", "缩容", "回滚", "日志", "监控", "配置", "安装", "迁移"],
    "concept": ["是什么", "原理", "概念", "定义", "架构", "设计", "机制", "工作原理", "如何实现"],
    "troubleshoot": ["报错", "故障", "异常", "失败", "排查", "诊断", "解决", "问题", "错误"],
    "optimization": ["优化", "性能", "瓶颈", "调优", "提升", "压测", "负载"],
}


def _detect_intent_legacy(query: str) -> str:
    """旧版关键词意图识别（兜底用）"""
    query_lower = query.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


class IntentChannel(RetrievalChannel):
    """意图定向检索通道 — 识别用户意图后路由到语义相似检索

    支持两套实现（由配置控制）：
    - 新版（默认）：使用 IntentResolver 并行解析，支持多问句拆分 + LLM 分类
    - 旧版（兜底）：纯关键词匹配，无并行，适合快速场景
    """

    name = "intent"

    # 懒加载 resolver
    _resolver: Any = None
    _rewrite_service: Any = None

    def _get_resolver(self):
        if self._resolver is None:
            from app.services.intent_resolver import get_intent_resolver
            self._resolver = get_intent_resolver()
        return self._resolver

    def _get_rewrite_service(self):
        if self._rewrite_service is None:
            from app.services.query_rewrite_service import get_query_rewrite_service
            self._rewrite_service = get_query_rewrite_service()
        return self._rewrite_service

    async def _retrieve_by_intent_filter(
        self, query: str, intent_id: str, top_k: int
    ) -> list[RetrievalResult]:
        """用意图 ID 作为 Milvus filter 进行检索"""
        from app.core.milvus_client import milvus_manager

        query_vector = vector_embedding_service.embed_query(query)
        collection = milvus_manager.get_collection()

        # 意图 ID 对应 Milvus metadata["intent_id"] 字段
        intent_expr = f'metadata["intent_id"] == "{intent_id}"'
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
                expr=intent_expr,
            )
        except Exception:
            # filter 字段不存在时降级为全局检索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
            )

        retrieval_results = []
        for hits in results:
            for hit in hits:
                retrieval_results.append(RetrievalResult(
                    doc_id=hit.entity.get("id"),
                    content=hit.entity.get("content", ""),
                    score=float(-hit.distance),
                    channel=self.name,
                    metadata={**hit.entity.get("metadata", {}), "intent_id": intent_id},
                ))
        return retrieval_results

    async def _retrieve_global(self, query: str, top_k: int) -> list[RetrievalResult]:
        """全局向量检索（降级兜底）"""
        from app.core.milvus_client import milvus_manager

        query_vector = vector_embedding_service.embed_query(query)
        collection = milvus_manager.get_collection()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "content", "metadata"],
        )
        retrieval_results = []
        for hits in results:
            for hit in hits:
                retrieval_results.append(RetrievalResult(
                    doc_id=hit.entity.get("id"),
                    content=hit.entity.get("content", ""),
                    score=float(-hit.distance),
                    channel=self.name,
                    metadata={**hit.entity.get("metadata", {}), "fallback": True},
                ))
        return retrieval_results

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            # Step1: 意图解析（并行，支持多子问题 + 路由决策）
            rewrite_service = self._get_rewrite_service()
            rewrite_result = await rewrite_service.rewrite_with_split(query)
            resolver = self._get_resolver()
            resolve_result = await resolver.resolve_async(rewrite_result)
            sub_intents = resolve_result.sub_intents

            if resolve_result.has_exception:
                logger.warning(f"[IntentChannel] 意图解析有异常（已降级兜底）: {resolve_result.exception_reason}")

            if not sub_intents:
                # 意图解析为空，降级全局检索
                logger.warning(f"[IntentChannel] 意图解析为空，降级全局检索: {query}")
                return await self._retrieve_global(query, top_k)

            # Step2: 多子问题并行检索（根据 routing_target 决定是否检索）
            all_tasks = []
            for sqi in sub_intents:
                # chitchat / clarification 不走 RAG 检索通道
                if sqi.routing_target in (
                    RoutingTarget.CHITCHAT,
                    RoutingTarget.CLARIFICATION,
                ):
                    continue
                for ns in sqi.node_scores:
                    all_tasks.append(
                        self._retrieve_by_intent_filter(sqi.sub_question, ns.node_id, top_k)
                    )

            if not all_tasks:
                return await self._retrieve_global(query, top_k)

            # asyncio.gather 并行执行所有子问题的意图检索
            channel_results: list[list[RetrievalResult]] = await asyncio.gather(
                *all_tasks, return_exceptions=True
            )

            # 收集有效结果，过滤异常
            retrieval_results: list[RetrievalResult] = []
            for i, result in enumerate(channel_results):
                if isinstance(result, Exception):
                    sqi_idx = i // max(len(sqi.node_scores), 1)
                    node_idx = i % max(len(sub_intents[min(sqi_idx, len(sub_intents)-1)].node_scores), 1)
                    logger.warning(
                        f"[IntentChannel] 意图检索异常: subQ={sqi_idx}, intent={node_idx}, err={result}"
                    )
                else:
                    retrieval_results.extend(result)

            # Step3: 无结果时降级全局检索
            if not retrieval_results:
                logger.debug(f"[IntentChannel] 所有意图检索无结果，降级全局检索")
                return await self._retrieve_global(query, top_k)

            logger.debug(
                f"[IntentChannel] 召回 {len(retrieval_results)} 条，"
                f"子问题={len(sub_intents)}，总意图={sum(len(sq.node_scores) for sq in sub_intents)}，"
                f"需二次确认={len([s for s in sub_intents if s.routing_target == RoutingTarget.CONFIRM_OPERATION])}"
            )
            return retrieval_results

        except Exception as e:
            logger.warning(f"[IntentChannel] 意图通道异常，降级全局检索: {e}")
            return await self._retrieve_global(query, top_k)


# ---------------------------------------------------------------------------
# 通道3: 关键词检索（BM25 思想）
# ---------------------------------------------------------------------------

class KeywordChannel(RetrievalChannel):
    """关键词检索通道 — 基于术语精确匹配的 BM25 思想

    当前使用 Milvus 过滤 + 全文匹配模拟。
    后续有 Elasticsearch 时改为 ES BM25 检索。
    """

    name = "keyword"

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        try:
            keywords = self._extract_keywords(query)
            if not keywords:
                logger.debug("[KeywordChannel] 未提取到关键词，跳过")
                return []

            logger.debug(f"[KeywordChannel] 关键词: {keywords}, 查询: {query}")

            # 用 OR 表达式过滤 metadata 中包含关键词的文档
            # Milvus 标量字段查询
            expressions = []
            for kw in keywords[:5]:  # 最多5个关键词
                safe_kw = kw.replace('"', '\\"')
                expressions.append(f'content like "%{safe_kw}%"')

            if not expressions:
                return []

            expr = " || ".join(expressions[:5])

            from app.core.milvus_client import milvus_manager
            collection = milvus_manager.get_collection()

            # 先用关键词过滤，再做向量相似度排序
            query_vector = vector_embedding_service.embed_query(query)

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "metadata"],
                expr=expr,
            )

            retrieval_results = []
            for hits in results:
                for hit in hits:
                    # BM25 思想：命中的关键词越多，得分越高
                    content = hit.entity.get("content", "")
                    hit_count = sum(kw in content for kw in keywords)
                    bm25_boost = hit_count / len(keywords)
                    final_score = float(-hit.distance) * (1 + bm25_boost)
                    retrieval_results.append(RetrievalResult(
                        doc_id=hit.entity.get("id"),
                        content=content,
                        score=final_score,
                        channel=self.name,
                        metadata=hit.entity.get("metadata", {}),
                    ))

            logger.debug(f"[KeywordChannel] 召回 {len(retrieval_results)} 条")
            return retrieval_results

        except Exception as e:
            logger.warning(f"[KeywordChannel] 检索失败: {e}")
            return []

    def _extract_keywords(self, query: str) -> list[str]:
        """从查询中提取关键词（去停用词）"""
        stopwords = {"的", "了", "是", "在", "和", "与", "对", "为", "如何", "怎么", "什么", "吗", "呢", "吧"}
        words = []
        for word in query:
            if word.strip() and word not in stopwords:
                words.append(word)
        # 返回连续2-4字的词组作为关键词
        keywords = []
        for length in [4, 3, 2]:
            for i in range(len(words) - length + 1):
                phrase = "".join(words[i:i + length])
                if phrase not in stopwords:
                    keywords.append(phrase)
        return keywords[:5]


# ---------------------------------------------------------------------------
# 后处理器
# ---------------------------------------------------------------------------

class PostProcessor(ABC):
    """后处理器抽象基类"""

    order: int = 0  # 执行顺序，数字越小越先执行

    @abstractmethod
    async def process(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        pass


class DeduplicationProcessor(PostProcessor):
    """去重处理器 — 基于 doc_id + 内容前64字哈希去重"""

    order = 1

    async def process(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        seen: dict[tuple[str, str], RetrievalResult] = {}
        for r in results:
            key = (r.doc_id, hashlib.md5(r.content[:64].encode()).hexdigest())
            if key not in seen:
                seen[key] = r
            else:
                # 保留最高分
                if r.score > seen[key].score:
                    seen[key] = r

        deduplicated = list(seen.values())
        dropped = len(results) - len(deduplicated)
        if dropped > 0:
            logger.debug(f"[DeduplicationProcessor] 去重: {len(results)} -> {len(deduplicated)} (移除 {dropped} 条重复)")
        return deduplicated


class RerankProcessor(PostProcessor):
    """重排序处理器 — 多通道得分加权后按综合分数重排

    当前策略: channel权重 × 原始score，再与 BM25 分数融合
    后续可替换为 Cross-Encoder (如 bge-reranker) 以提升精度
    """

    order = 10

    # 通道权重配置
    CHANNEL_WEIGHTS = {
        "vector": 0.5,
        "intent": 0.3,
        "keyword": 0.2,
    }

    async def process(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        if not results:
            return results

        # Step1: 归一化各通道得分（Min-Max 到 [0,1]）
        scores_by_channel: dict[str, list[float]] = {}
        for r in results:
            scores_by_channel.setdefault(r.channel, []).append(r.score)

        normalized_scores: dict[RetrievalResult, float] = {}
        for r in results:
            channel_scores = scores_by_channel[r.channel]
            if len(channel_scores) > 1:
                min_s, max_s = min(channel_scores), max(channel_scores)
                if max_s > min_s:
                    normalized = (r.score - min_s) / (max_s - min_s)
                else:
                    normalized = 1.0
            else:
                normalized = 1.0

            # Step2: 加权融合
            weight = self.CHANNEL_WEIGHTS.get(r.channel, 0.2)
            normalized_scores[r] = normalized * weight

        # Step3: 跨通道综合排序
        reranked = sorted(results, key=lambda r: normalized_scores[r], reverse=True)

        logger.debug(f"[RerankProcessor] 重排: {len(reranked)} 条，Top1 doc_id={reranked[0].doc_id}")
        return reranked


class PostProcessorPipeline:
    """后处理流水线 — 按 order 顺序执行各处理器"""

    def __init__(self):
        self.processors: list[PostProcessor] = []

    def add_processor(self, processor: PostProcessor):
        self.processors.append(processor)
        self.processors.sort(key=lambda p: p.order)

    async def execute(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> list[RetrievalResult]:
        current = results
        for processor in self.processors:
            current = await processor.process(current, query)
        return current


# ---------------------------------------------------------------------------
# 多通道检索引擎入口
# ---------------------------------------------------------------------------

class MultiChannelRetrievalEngine:
    """多通道检索引擎 — 并行调度 + 后处理流水线"""

    def __init__(self):
        self.channels: list[RetrievalChannel] = [
            VectorChannel(),
            IntentChannel(),
            KeywordChannel(),
        ]
        self.pipeline = PostProcessorPipeline()
        self.pipeline.add_processor(DeduplicationProcessor())
        self.pipeline.add_processor(RerankProcessor())

        # 每通道召回数量（预留给后处理去重截断）
        self.channel_top_k = config.rag_top_k * 4

        logger.info(f"多通道检索引擎初始化完成，共 {len(self.channels)} 个通道")

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """
        执行多通道并行检索

        Args:
            query: 用户查询
            top_k: 最终返回数量（None 时使用 config.rag_top_k）

        Returns:
            list[RetrievalResult]: 检索结果（已去重、重排）
        """
        if top_k is None:
            top_k = config.rag_top_k

        logger.info(f"[MultiChannelEngine] 收到查询: '{query}', top_k={top_k}")

        # Step1: 并行检索
        tasks = [
            channel.retrieve(query, top_k=self.channel_top_k)
            for channel in self.channels
        ]
        channel_results: list[list[RetrievalResult]] = await asyncio.gather(*tasks)

        # Step2: 合并候选池
        all_candidates: list[RetrievalResult] = []
        for i, results in enumerate(channel_results):
            all_candidates.extend(results)
            logger.debug(
                f"[MultiChannelEngine] {self.channels[i].name} 通道召回 {len(results)} 条"
            )

        logger.info(
            f"[MultiChannelEngine] 并行检索完成，候选总数: {len(all_candidates)} 条"
        )

        # Step3: 后处理流水线
        processed = await self.pipeline.execute(all_candidates, query)

        # Step4: 截断到 top_k
        final_results = processed[:top_k]

        channel_counts = {ch.name: sum(1 for r in final_results if r.channel == ch.name)
                          for ch in self.channels}
        logger.info(
            f"[MultiChannelEngine] 最终返回 {len(final_results)} 条，"
            f"各通道贡献: {channel_counts}"
        )

        return final_results


# 全局单例（延迟初始化）
_retrieval_engine: MultiChannelRetrievalEngine | None = None


def get_retrieval_engine() -> MultiChannelRetrievalEngine:
    global _retrieval_engine
    if _retrieval_engine is None:
        _retrieval_engine = MultiChannelRetrievalEngine()
    return _retrieval_engine
