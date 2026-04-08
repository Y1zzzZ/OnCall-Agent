"""问句改写与子问题拆分服务

功能：
1. 规范化输入（去噪音、全角转半角、修正拼写）
2. 检测是否为多问句（LLM 方式 or 规则方式）
3. 若为多问句，拆分成多个子问题分别处理
4. 可选：LLM 改写问句（同义词替换、补充省略主语）

默认行为：
- 多问句拆分默认关闭（rule_based_split 兜底）
- LLM 改写默认关闭
- 通过配置项 query_rewrite_enabled 和 multi_question_split_enabled 控制
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from app.config import config


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RewriteResult:
    """问句改写与拆分结果"""
    original_question: str          # 用户原始输入
    rewritten_question: str         # 规范化/改写后的单一问句（无多问句时与 original 相同）
    sub_questions: list[str] = field(
        default_factory=list
    )  # 拆分的子问题列表，若为空则表示无需拆分

    @property
    def is_multi_question(self) -> bool:
        """是否是多问句（需要拆分处理）"""
        return len(self.sub_questions) > 1

    @property
    def questions_to_process(self) -> list[str]:
        """实际需要处理的问句列表（子问题列表，若为空则处理改写后问句）"""
        return self.sub_questions if self.is_multi_question else [self.rewritten_question]


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------

def _normalize_question(question: str) -> str:
    """规范化问句：去首尾空格、合并多余空格"""
    question = question.strip()
    question = re.sub(r"\s+", " ", question)
    return question


# ---------------------------------------------------------------------------
# 规则拆分（轻量兜底，无需 LLM 调用）
# ---------------------------------------------------------------------------

def rule_based_split(question: str) -> list[str]:
    """基于标点符号和连接词的规则拆分

    拆分点：？ 。 ； 同时 另外 并且 以及 或者 和 与 再加 再问
    保留拆分后的每个独立问句。
    """

    normalized = _normalize_question(question)

    # 按中英文标点和特定连接词切分
    # 保留分隔符，便于判断是否为有效问句
    split_pattern = r"[？?。\n]|(?:\s+(?:同时|另外|并且|以及|或者|和|与|再加|再问|另外|还有|并且|再说)\s*)"
    parts = re.split(split_pattern, normalized)

    sub_questions = []
    for part in parts:
        part = part.strip()
        # 过滤掉空片段和纯停用词片段
        if part and len(part) >= 4:
            sub_questions.append(part)

    # 如果拆分后只有1个或0个，保持原样
    if len(sub_questions) <= 1:
        return [normalized] if normalized else []

    logger.debug(f"[QueryRewrite] 规则拆分: {normalized[:40]} → {sub_questions}")
    return sub_questions


# ---------------------------------------------------------------------------
# LLM 改写与智能拆分
# ---------------------------------------------------------------------------

def _build_llm_rewrite_prompt(question: str) -> tuple[str, str]:
    """构建 LLM 改写 + 拆分的 prompt"""

    system_prompt = """你是一个查询改写助手。用户可能输入多问句、口语化表达或不完整的问句。

你的任务：
1. 将问句改写为规范、完整、清晰的中文问句
2. 如果用户同时问了多个问题，将其拆分为独立的子问题

输出格式（必须严格遵守，直接输出 JSON，不要解释）：
{
  "rewritten_question": "改写后的单一完整问句，若无需改写则保持原样",
  "sub_questions": ["子问题1", "子问题2"]   // 若为单问句则为空数组[]
}

拆分原则：
- 用户明确用"？"、"。"、"同时"、"另外"等分隔多个独立问题时，必须拆分
- 用户连续用"和"、"与"、"以及"连接多个相关但独立的问题时，拆分
- 用户问的是一个问题的不同方面，不要拆分（如"K8s怎么部署和扩缩容"算一个整体问题）

注意事项：
- 直接输出 JSON，不要 Markdown 包裹，不要解释
- rewritten_question 和 sub_questions 字段必须存在
- sub_questions 为空时表示无需拆分"""

    return system_prompt, question


def _parse_rewrite_response(raw: str) -> dict:
    """解析 LLM 返回的改写结果"""
    import json
    try:
        cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"[QueryRewrite] LLM 返回 JSON 解析失败: {e}, raw: {raw[:100]}")
        return {"rewritten_question": raw, "sub_questions": []}


async def _call_llm_rewrite(question: str) -> RewriteResult:
    """调用 LLM 进行问句改写与拆分"""
    from app.core.llm_factory import llm_factory
    from app.config import config

    model = llm_factory.create_chat_model(
        model=config.rag_model,
        temperature=0.1,
        streaming=False,
    )

    system_prompt, user_prompt = _build_llm_rewrite_prompt(question)

    try:
        response = model.invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[QueryRewrite] LLM 改写调用失败: {e}")
        return RewriteResult(
            original_question=question,
            rewritten_question=question,
            sub_questions=[],
        )

    parsed = _parse_rewrite_response(raw_text)
    rewritten = parsed.get("rewritten_question", question)
    sub_questions: list[str] = parsed.get("sub_questions", [])

    # 如果 LLM 返回空数组或非列表，降级为规则拆分
    if not isinstance(sub_questions, list) or len(sub_questions) == 0:
        sub_questions = rule_based_split(question)
        rewritten = question

    return RewriteResult(
        original_question=question,
        rewritten_question=rewritten,
        sub_questions=sub_questions,
    )


# ---------------------------------------------------------------------------
# 服务主类
# ---------------------------------------------------------------------------

class QueryRewriteService:
    """问句改写与拆分服务

    行为由配置控制：
    - query_rewrite_enabled: 是否启用 LLM 改写（默认 False）
    - multi_question_split_enabled: 是否启用多问句拆分（默认 False，开启时自动启用规则拆分）

    若拆分关闭：始终返回 RewriteResult(sub_questions=[])，引擎走单问句逻辑
    若拆分开启 + 改写关闭：使用规则拆分
    若拆分开启 + 改写开启：使用 LLM 智能拆分
    """

    def __init__(self):
        self.split_enabled: bool = getattr(config, "multi_question_split_enabled", False)
        self.rewrite_enabled: bool = getattr(config, "query_rewrite_enabled", False)
        logger.info(
            f"[QueryRewrite] 初始化: 多问句拆分={'开启' if self.split_enabled else '关闭'}, "
            f"LLM改写={'开启' if self.rewrite_enabled else '关闭'}"
        )

    async def rewrite_with_split(self, question: str) -> RewriteResult:
        """执行改写 + 拆分

        Args:
            question: 用户原始问句

        Returns:
            RewriteResult，含 original_question、rewritten_question、sub_questions
        """
        question = _normalize_question(question)

        if not self.split_enabled:
            # 拆分关闭，走原有单问句逻辑
            return RewriteResult(
                original_question=question,
                rewritten_question=question,
                sub_questions=[],
            )

        if self.rewrite_enabled:
            # LLM 改写 + 智能拆分
            result = await _call_llm_rewrite(question)
            logger.debug(
                f"[QueryRewrite] LLM处理: {question[:30]}... → "
                f"子问题数={len(result.sub_questions)}"
            )
            return result
        else:
            # 规则拆分
            subs = rule_based_split(question)
            return RewriteResult(
                original_question=question,
                rewritten_question=question,
                sub_questions=subs if len(subs) > 1 else [],
            )

    def rewrite_with_split_sync(self, question: str) -> RewriteResult:
        """同步版本（仅规则拆分，无 LLM 调用）"""
        question = _normalize_question(question)

        if not self.split_enabled:
            return RewriteResult(
                original_question=question,
                rewritten_question=question,
                sub_questions=[],
            )

        subs = rule_based_split(question)
        return RewriteResult(
            original_question=question,
            rewritten_question=question,
            sub_questions=subs if len(subs) > 1 else [],
        )


# 全局单例
_query_rewrite_service: QueryRewriteService | None = None


def get_query_rewrite_service() -> QueryRewriteService:
    global _query_rewrite_service
    if _query_rewrite_service is None:
        _query_rewrite_service = QueryRewriteService()
    return _query_rewrite_service
