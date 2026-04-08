"""并行意图解析器

三层架构：
1. 预过滤层（intent_prefilter）：极低成本快速过滤 chitchat / clarification
2. 意图树打分层（LLM / 关键词）：对业务问题打多意图分数
3. 路由决策层：根据 category 决定下游走 RAG / 工具调用 / 澄清引导

降级策略（保守兜底）：
- 预过滤层异常 → 继续进入打分层
- LLM 调用异常 → 降级关键词分类器
- 关键词分类也失败 → 走 knowledge（RAG），不让系统崩溃
- 所有异常默认掰到 knowledge，RAG 最多浪费一次检索，不会死循环

线程池：
- 使用 Python ThreadPoolExecutor 模拟 Java intentClassifyThreadPoolExecutor
- 意图分类是 IO 密集型（LLM 调用），适合多线程并行
- 核心线程数默认取 CPU 核数 × 2，上限 8
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from app.config import config
from app.services.intent_classifier import (
    BaseIntentClassifier,
    NodeScore,
    get_intent_classifier,
)
from app.services.intent_prefilter import PrefilterIntent, prefilter
from app.services.query_rewrite_service import RewriteResult


# ---------------------------------------------------------------------------
# 路由目标枚举（第三层：下游决定走哪条管道）
# ---------------------------------------------------------------------------

class RoutingTarget(Enum):
    """意图路由目标，决定下游走哪条处理管道"""
    RAG = "rag"                          # 走 RAG 检索（concept 类意图，或无高分 operation 时兜底）
    CONFIRM_OPERATION = "confirm_op"     # operation 类意图高置信，需二次确认是查文档还是执行工具
    CLARIFICATION = "clarification"      # 意图模糊，需要反问引导用户
    CHITCHAT = "chitchat"                # 纯闲聊，直接用 reply_text 回复用户
    FALLBACK_KNOWLEDGE = "fallback_knowledge"  # 兜底：异常 / 无意图 / 低于阈值 → 走 knowledge


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SubQuestionIntent:
    """子问题 + 其识别出的意图列表 + 路由决策"""
    sub_question: str
    node_scores: list[NodeScore] = field(default_factory=list)
    # 元信息
    fallback: bool = False               # 是否降级使用了关键词分类器
    # 预过滤结果（如果该子问题在预过滤层被命中）
    prefilter_result: Optional[PrefilterIntent] = None
    # 路由决策：根据 top intent 的 category 和分数决定
    routing_target: RoutingTarget = RoutingTarget.RAG

    @property
    def top_intent_id(self) -> str | None:
        return self.node_scores[0].node_id if self.node_scores else None

    @property
    def has_operation_intent(self) -> bool:
        """是否有 operation 类意图（可能需要工具调用）"""
        return any(ns.category == "operation" for ns in self.node_scores)

    @property
    def top_score(self) -> float:
        return self.node_scores[0].score if self.node_scores else 0.0


@dataclass
class IntentResolveResult:
    """完整的意图解析结果"""
    # 每个子问题的解析结果（顺序与 rewrite_result.questions_to_process 一致）
    sub_intents: list[SubQuestionIntent]
    # 全局是否发生了异常（用于调用方判断是否需要告警）
    has_exception: bool = False
    exception_reason: str = ""


# ---------------------------------------------------------------------------
# 意图数量上限配置
# ---------------------------------------------------------------------------

# 单个子问题的最大意图数量
MAX_INTENT_PER_SUBQUESTION = getattr(config, "max_intent_per_subquestion", 5)
# 所有子问题的意图总上限
MAX_TOTAL_INTENTS = getattr(config, "max_total_intents", 10)
# 意图分数下限（低于此分过滤掉）
INTENT_MIN_SCORE = getattr(config, "intent_min_score", 0.3)
# operation 类意图需要多高才触发 CONFIRM_OPERATION（防止误触发工具）
OPERATION_CONFIRM_THRESHOLD = getattr(config, "operation_confirm_threshold", 0.7)


# ---------------------------------------------------------------------------
# 意图候选收集（用于上限控制）
# ---------------------------------------------------------------------------

@dataclass
class IntentCandidate:
    """跨子问题的意图候选（扁平化结构，方便排序）"""
    sub_question: str
    node_score: NodeScore
    subquestion_index: int   # 子问题在原始列表中的索引，用于保底策略


# ---------------------------------------------------------------------------
# 上限控制逻辑
# ---------------------------------------------------------------------------

def _cap_total_intents(
    sub_intents: list[SubQuestionIntent],
    max_total: int = MAX_TOTAL_INTENTS,
    max_per_sub: int = MAX_INTENT_PER_SUBQUESTION,
    min_score: float = INTENT_MIN_SCORE,
) -> list[SubQuestionIntent]:
    """限制意图总数量，每个子问题至少保留 1 个最高分意图

    策略：
    1. 过滤低分意图（< min_score）
    2. 统计总意图数，若未超限直接返回
    3. 每个子问题先保底 1 个最高分意图
    4. 剩余配额按分数从高到低分配（跨子问题竞争）
    """

    # Step 1: 过滤低分 + 限制单子问题上限
    filtered: list[SubQuestionIntent] = []
    for sq in sub_intents:
        filtered_scores = [
            ns for ns in sq.node_scores
            if ns.score >= min_score
        ][:max_per_sub]
        filtered.append(SubQuestionIntent(
            sub_question=sq.sub_question,
            node_scores=filtered_scores,
            fallback=sq.fallback,
            prefilter_result=sq.prefilter_result,
            routing_target=sq.routing_target,
        ))

    # Step 2: 统计总意图数
    total_intents = sum(len(sq.node_scores) for sq in filtered)
    if total_intents <= max_total:
        return filtered

    # Step 3: 收集所有候选，打平
    all_candidates: list[IntentCandidate] = []
    for sqi, sq in enumerate(filtered):
        for ns in sq.node_scores:
            all_candidates.append(IntentCandidate(
                sub_question=sq.sub_question,
                node_score=ns,
                subquestion_index=sqi,
            ))

    # Step 4: 每个子问题保底 1 个（取最高分）
    guaranteed_ids: set[int] = set()  # 已保底的 subquestion_index
    remaining_budget = max_total

    # 按分数降序排序候选
    sorted_candidates = sorted(all_candidates, key=lambda c: c.node_score.score, reverse=True)

    # 第一轮：保底（每个子问题取最高分）
    for candidate in sorted_candidates:
        if len(guaranteed_ids) >= len(filtered):
            break   # 所有子问题都已保底
        if candidate.subquestion_index not in guaranteed_ids:
            guaranteed_ids.add(candidate.subquestion_index)
            remaining_budget -= 1

    if remaining_budget <= 0:
        # 配额仅够保底，每个子问题只保留1个
        for sqi, sq in enumerate(filtered):
            sq.node_scores = [sq.node_scores[0]] if sq.node_scores else []
        return filtered

    # 第二轮：剩余配额按分数从高到低分配
    additional_ids: set[int] = set()
    for candidate in sorted_candidates:
        if remaining_budget <= 0:
            break
        if candidate.subquestion_index not in guaranteed_ids and candidate.subquestion_index not in additional_ids:
            additional_ids.add(candidate.subquestion_index)
            remaining_budget -= 1

    # Step 5: 重建结果
    final_sub_intents: list[SubQuestionIntent] = []
    for sqi, sq in enumerate(filtered):
        kept: list[NodeScore] = []
        for candidate in sorted_candidates:
            if candidate.subquestion_index == sqi:
                if sqi in guaranteed_ids or sqi in additional_ids:
                    kept.append(candidate.node_score)
        kept = sorted(kept, key=lambda ns: ns.score, reverse=True)
        kept = kept[:max_per_sub]
        final_sub_intents.append(SubQuestionIntent(
            sub_question=sq.sub_question,
            node_scores=kept,
            fallback=sq.fallback,
            prefilter_result=sq.prefilter_result,
            routing_target=sq.routing_target,
        ))

    return final_sub_intents


# ---------------------------------------------------------------------------
# 路由决策逻辑（第三层）
# ---------------------------------------------------------------------------

def _decide_routing(sqi: SubQuestionIntent) -> RoutingTarget:
    """根据子问题的意图打分结果，决定下游路由目标

    决策规则：
    1. 预过滤已命中 chitchat → CHITCHAT
    2. 预过滤已命中 clarification → CLARIFICATION
    3. 没有有效意图（空列表）→ 兜底走 knowledge（RAG）
    4. 所有意图分数都低于阈值 → 兜底走 knowledge（RAG）
    5. Top1 是 operation 且分数 >= OPERATION_CONFIRM_THRESHOLD → CONFIRM_OPERATION
    6. 否则 → RAG（走 RAG 检索）
    """
    # 1. 预过滤 chitchat
    if sqi.prefilter_result == PrefilterIntent.CHITCHAT:
        return RoutingTarget.CHITCHAT

    # 2. 预过滤 clarification
    if sqi.prefilter_result == PrefilterIntent.CLARIFICATION:
        return RoutingTarget.CLARIFICATION

    # 3. 无有效意图
    if not sqi.node_scores:
        return RoutingTarget.FALLBACK_KNOWLEDGE

    top = sqi.node_scores[0]

    # 4. 低于最低阈值 → 兜底 knowledge
    if top.score < INTENT_MIN_SCORE:
        return RoutingTarget.FALLBACK_KNOWLEDGE

    # 5. operation 类意图高置信 → 需要二次确认
    if top.category == "operation" and top.score >= OPERATION_CONFIRM_THRESHOLD:
        return RoutingTarget.CONFIRM_OPERATION

    # 6. 默认走 RAG
    return RoutingTarget.RAG


def _apply_routing(sub_intents: list[SubQuestionIntent]) -> list[SubQuestionIntent]:
    """对所有子问题应用路由决策"""
    for sqi in sub_intents:
        sqi.routing_target = _decide_routing(sqi)
    return sub_intents


# ---------------------------------------------------------------------------
# 并行解析器
# ---------------------------------------------------------------------------

class IntentResolver:
    """并行意图解析器

    三层处理流程：
    第一层：预过滤（intent_prefilter），快速规则过滤 chitchat / clarification
    第二层：意图树打分类（LLM / 关键词），对业务问题打分
    第三层：路由决策，根据 category 决定下游管道

    支持两种分类器：
    - LLMIntentClassifier: 调用大模型对所有叶子意图打分（精确但慢）
    - KeywordIntentClassifier: 关键词匹配打分（快速，无 LLM 调用）

    默认优先用 LLM 分类器；LLM 不可用时自动降级到关键词分类器。
    """

    def __init__(
        self,
        classifier: BaseIntentClassifier | None = None,
        max_workers: int | None = None,
    ):
        # 意图分类器
        self._classifier = classifier or get_intent_classifier()
        # 线程池大小（默认 CPU 核数 × 2，上限 8）
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count * 2, 8)
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="intent_classify_",
            )
        logger.debug(f"[IntentResolver] 线程池初始化，max_workers={self._max_workers}")
        return self._executor

    # ---------------------------------------------------------------------------
    # 同步入口（ThreadPoolExecutor 并行）
    # ---------------------------------------------------------------------------

    def resolve(self, rewrite_result: RewriteResult) -> IntentResolveResult:
        """同步并行解析（线程池），用于 async 环境中的同步子任务

        Args:
            rewrite_result: 问句改写与拆分结果

        Returns:
            IntentResolveResult，包含每个子问题的意图列表、路由决策和异常状态
        """
        questions = rewrite_result.questions_to_process
        has_exception = False
        exception_reason = ""

        if len(questions) == 1:
            # 单问句不需要并行，直接调用
            sqi = self._process_single(questions[0])
            result = [sqi]
        else:
            # 多问句：线程池并行
            executor = self._get_executor()
            futures = {
                executor.submit(self._process_single, q): q
                for q in questions
            }
            results: list[SubQuestionIntent] = []
            for future in as_completed(futures):
                try:
                    sqi = future.result()
                    results.append(sqi)
                except Exception as e:
                    q = futures[future]
                    has_exception = True
                    exception_reason = str(e)
                    logger.error(f"[IntentResolver] 子问题意图识别失败: {q}, 错误: {e}")
                    # 兜底：失败则走 knowledge
                    results.append(SubQuestionIntent(
                        sub_question=q,
                        node_scores=[],
                        routing_target=RoutingTarget.FALLBACK_KNOWLEDGE,
                    ))

            # 按原始顺序重排（as_completed 不保证顺序）
            order_map = {q: i for i, q in enumerate(questions)}
            results.sort(key=lambda sqi: order_map.get(sqi.sub_question, 999))

        # 上限控制
        capped = _cap_total_intents(
            results,
            max_total=MAX_TOTAL_INTENTS,
            max_per_sub=MAX_INTENT_PER_SUBQUESTION,
            min_score=INTENT_MIN_SCORE,
        )

        # 路由决策（第三层）
        final = _apply_routing(capped)

        total = sum(len(sqi.node_scores) for sqi in final)
        logger.info(
            f"[IntentResolver] 解析完成: {len(questions)} 个子问题 → "
            f"{total} 个有效意图，{len([s for s in final if s.routing_target == RoutingTarget.CONFIRM_OPERATION])} 个需二次确认"
        )

        return IntentResolveResult(
            sub_intents=final,
            has_exception=has_exception,
            exception_reason=exception_reason,
        )

    def _process_single(self, question: str) -> SubQuestionIntent:
        """对一个子问题完成三层处理：预过滤 → 分类 → 路由决策"""
        sqi = SubQuestionIntent(sub_question=question)

        # ---------- 第一层：预过滤 ----------
        try:
            pf_result = prefilter(question)
            if pf_result is not None:
                sqi.prefilter_result = pf_result.intent
                logger.debug(f"[IntentResolver] 预过滤命中: {question[:30]!r} → {pf_result.intent.value}")
                # 预过滤命中时，不调用 LLM，直接完成（路由由 decide_routing 决定）
                sqi.routing_target = _decide_routing(sqi)
                return sqi
        except Exception as e:
            # 预过滤异常不算严重，继续进入分类层
            logger.warning(f"[IntentResolver] 预过滤异常，继续进入分类层: {e}")

        # ---------- 第二层：意图分类 ----------
        try:
            scores = self._classifier.classify(question)
            logger.debug(f"[IntentResolver] 分类完成: {question[:30]}... → {len(scores)} 个候选意图")
            sqi.node_scores = scores
        except Exception as e:
            logger.warning(f"[IntentResolver] LLM分类失败，降级为关键词分类: {e}")
            # 降级到关键词分类器
            from app.services.intent_classifier import KeywordIntentClassifier
            fallback_clf = KeywordIntentClassifier()
            try:
                scores = fallback_clf.classify(question)
                sqi.node_scores = scores
                sqi.fallback = True
                logger.info(f"[IntentResolver] 关键词分类兜底成功: {question[:30]}... → {len(scores)} 个意图")
            except Exception as e2:
                logger.error(f"[IntentResolver] 关键词分类也失败: {e2}")
                # 兜底：所有异常都走 knowledge（RAG），不让系统崩溃
                sqi.node_scores = []
                sqi.routing_target = RoutingTarget.FALLBACK_KNOWLEDGE
                return sqi

        # ---------- 第三层：路由决策 ----------
        sqi.routing_target = _decide_routing(sqi)
        return sqi

    # ---------------------------------------------------------------------------
    # 异步入口（asyncio 并行，兼容 async/await）
    # ---------------------------------------------------------------------------

    async def resolve_async(self, rewrite_result: RewriteResult) -> IntentResolveResult:
        """异步并行解析（使用 asyncio.to_thread 包装线程池任务）

        用于 async 环境（如 FastAPI 路由）与原有 async 代码无缝集成。
        """
        loop = asyncio.get_running_loop()
        questions = rewrite_result.questions_to_process
        has_exception = False
        exception_reason = ""

        if len(questions) == 1:
            # 单问句
            sqi = await loop.run_in_executor(None, self._process_single, questions[0])
            result = [sqi]
        else:
            # 多问句：并行调度
            tasks = [
                loop.run_in_executor(None, self._process_single, q)
                for q in questions
            ]
            results: list[SubQuestionIntent] = []
            done, pending = await asyncio.wait(tasks)
            for task in done:
                try:
                    results.append(task.result())
                except Exception as e:
                    has_exception = True
                    exception_reason = str(e)
                    logger.error(f"[IntentResolver] 异步意图识别失败: {e}")
                    # 兜底：失败则走 knowledge
                    results.append(SubQuestionIntent(
                        sub_question="<unknown>",
                        node_scores=[],
                        routing_target=RoutingTarget.FALLBACK_KNOWLEDGE,
                    ))
            # 按原始顺序重排
            order_map = {q: i for i, q in enumerate(questions)}
            results.sort(key=lambda sqi: order_map.get(sqi.sub_question, 999))

        # 上限控制
        capped = _cap_total_intents(
            results,
            max_total=MAX_TOTAL_INTENTS,
            max_per_sub=MAX_INTENT_PER_SUBQUESTION,
            min_score=INTENT_MIN_SCORE,
        )

        # 路由决策
        final = _apply_routing(capped)

        return IntentResolveResult(
            sub_intents=final,
            has_exception=has_exception,
            exception_reason=exception_reason,
        )


# ---------------------------------------------------------------------------
# 全局单例（延迟初始化）
# ---------------------------------------------------------------------------

_intent_resolver: IntentResolver | None = None


def get_intent_resolver() -> IntentResolver:
    global _intent_resolver
    if _intent_resolver is None:
        _intent_resolver = IntentResolver()
    return _intent_resolver
