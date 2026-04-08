"""意图预过滤层 —— chitchat / clarification 快速规则命中

第一层防护：极低成本（无 LLM 调用），在进入意图树打分层之前，
用正则 + 关键词白名单过滤掉明显不属于业务意图的输入。

设计原则：
- 匹配则立即返回，不进入后续 LLM 打分
- 未匹配则继续进入 intent_resolver 的 LLM 打分层
- chitchat 关键词不区分大小写，clarification 按长度 + 关键词联合判断
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re

from loguru import logger


# ---------------------------------------------------------------------------
# 预过滤结果枚举
# ---------------------------------------------------------------------------

class PrefilterIntent(Enum):
    """预过滤层识别出的特殊意图类型"""
    CHITCHAT = "chitchat"            # 纯闲聊客套话
    CLARIFICATION = "clarification"  # 需要引导澄清的模糊问题
    # NOTE: "knowledge" / "operation" / "tool" 等业务意图不在这里处理，
    #       统一由 intent_resolver 的 LLM 打分层决定。
    UNKNOWN = "unknown"              # 无法确定，需要进入 LLM 打分层


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PrefilterResult:
    """预过滤结果"""
    intent: PrefilterIntent
    matched_text: str = ""          # 命中的关键词（用于日志 / 调试）
    reply_text: str = ""             # 如果是 chitchat/clarification，LLM 应回复的内容
    confidence: float = 1.0          # 规则匹配置信度（1.0 = 规则层直接确认）


# ---------------------------------------------------------------------------
# 规则定义
# ---------------------------------------------------------------------------

# chitchat 白名单（大小写不敏感，支持子串匹配）
_CHITCHAT_PATTERNS: list[re.Pattern] = [
    # 问候语
    re.compile(r"^你好[啊呀么嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^您好[啊呀么嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^hi\s*", re.IGNORECASE),
    re.compile(r"^hello\s*", re.IGNORECASE),
    re.compile(r"^hey\s*", re.IGNORECASE),
    # 感谢语
    re.compile(r"^谢谢[你您我啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^感谢[你您我啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^多谢[你您我啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^thanks?\s*", re.IGNORECASE),
    re.compile(r"^thx\s*", re.IGNORECASE),
    # 确认/同意
    re.compile(r"^好的[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^嗯嗯?[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^嗯[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^行[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^收到[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^OK\.?\s*$", re.IGNORECASE),
    re.compile(r"^ok\.?\s*$", re.IGNORECASE),
    re.compile(r"^yes\.?\s*$", re.IGNORECASE),
    re.compile(r"^yep\.?\s*$", re.IGNORECASE),
    # 道别语
    re.compile(r"^再见[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^拜拜[啊呀嘛。！？]*$", re.IGNORECASE),
    re.compile(r"^bye\.?\s*$", re.IGNORECASE),
    # 纯符号 / 无意义输入
    re.compile(r"^[，。！？、；：""''（）【】《》～·……—\s]+$"),
    re.compile(r"^[?!]+$"),
    re.compile(r"^\.\.+$"),
    # 在不在（纯问在不在，无其他内容）
    re.compile(r"^在[吗嘛么]?$", re.IGNORECASE),
    re.compile(r"^在不在$", re.IGNORECASE),
    re.compile(r"^在不$", re.IGNORECASE),
]

# clarification 关键词（长度极短时配合关键词判断）
_CLARIFICATION_KEYWORDS: list[re.Pattern] = [
    re.compile(r"^怎么办[？?。]*$", re.IGNORECASE),
    re.compile(r"^怎么办啊[？?。]*$", re.IGNORECASE),
    re.compile(r"^怎么处理[？?。]*$", re.IGNORECASE),
    re.compile(r"^怎么处理啊[？?。]*$", re.IGNORECASE),
    re.compile(r"^那件事$", re.IGNORECASE),
    re.compile(r"^那东西$", re.IGNORECASE),
    re.compile(r"^那个事情$", re.IGNORECASE),
    re.compile(r"^你说呢[？?。]*$", re.IGNORECASE),
    re.compile(r"^还有吗[？?。]*$", re.IGNORECASE),
    re.compile(r"^然后呢[？?。]*$", re.IGNORECASE),
    re.compile(r"^接下来呢[？?。]*$", re.IGNORECASE),
    re.compile(r"^所以呢[？?。]*$", re.IGNORECASE),
    re.compile(r"^嗯[。 ]+$", re.IGNORECASE),
]

# clarification 长度阈值：超过此长度的问题即使有歧义也不在规则层拦截
_CLARIFICATION_MAX_LEN = 8


# ---------------------------------------------------------------------------
# 预过滤逻辑
# ---------------------------------------------------------------------------

def prefilter(question: str) -> Optional[PrefilterResult]:
    """对用户输入进行快速预过滤，返回 PrefilterResult 或 None

    Args:
        question: 用户原始输入（已去除首尾空白）

    Returns:
        PrefilterResult if matched a special pattern (chitchat / clarification),
        None if the question should continue to the LLM scoring layer.

    规则优先级：
        1. chitchat 白名单正则 → 直接返回 chitchat
        2. clarification 关键词 + 长度阈值 → 返回 clarification
        3. 其余全部 → None（进入 LLM 打分层）
    """

    q = question.strip()
    if not q:
        return PrefilterResult(
            intent=PrefilterIntent.CHITCHAT,
            matched_text="<empty>",
            reply_text="您好！有什么可以帮您的？",
            confidence=1.0,
        )

    # ---------- 1. chitchat 匹配 ----------
    for pattern in _CHITCHAT_PATTERNS:
        if pattern.search(q):
            logger.debug(f"[Prefilter] chitchat 命中: {q!r} → {pattern.pattern!r}")
            return PrefilterResult(
                intent=PrefilterIntent.CHITCHAT,
                matched_text=pattern.pattern,
                reply_text=_get_chitchat_reply(q),
                confidence=1.0,
            )

    # ---------- 2. clarification 匹配 ----------
    # 短句 + clarification 关键词才命中（防止误伤正常短问句）
    if len(q) <= _CLARIFICATION_MAX_LEN:
        for pattern in _CLARIFICATION_KEYWORDS:
            if pattern.search(q):
                logger.debug(f"[Prefilter] clarification 命中: {q!r} → {pattern.pattern!r}")
                return PrefilterResult(
                    intent=PrefilterIntent.CLARIFICATION,
                    matched_text=pattern.pattern,
                    reply_text="您是想了解哪方面的情况呢？可以告诉我具体的问题或系统名称，我会尽力帮您解答。",
                    confidence=1.0,
                )

    # ---------- 3. 未命中，返回 None（继续进入 LLM 打分层）----------
    return None


def _get_chitchat_reply(question: str) -> str:
    """根据 chitchat 内容返回合适的回复模板"""
    q_lower = question.strip().lower()

    if any(k in q_lower for k in ["你好", "您好", "hi", "hello", "hey"]):
        return "您好！有什么可以帮您的？"
    if any(k in q_lower for k in ["谢谢", "感谢", "多谢", "thanks", "thx"]):
        return "不客气！请问还有其他问题吗？"
    if any(k in q_lower for k in ["好的", "嗯嗯", "嗯", "行", "收到", "ok", "yes", "yep"]):
        return "好的，还有什么需要帮忙的吗？"
    if any(k in q_lower for k in ["再见", "拜拜", "bye"]):
        return "再见！祝您工作顺利，有需要随时找我。"
    if any(k in q_lower for k in ["在", "在不在"]):
        return "在的！请问有什么问题需要我帮忙？"

    # 默认通用回复
    return "请问有什么我可以帮您的？"
