"""LLM 意图分类器 - 对用户问句进行意图识别和打分

核心逻辑：
1. 将所有叶子意图节点构造成 prompt，输入用户问句
2. 调用 LLM 对所有叶子节点打分（0~1），返回 JSON
3. 解析 JSON，返回排序后的 NodeScore 列表
4. 分数低于 INTENT_MIN_SCORE 的过滤掉，数量限制由调用方控制
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.config import config
from app.core.llm_factory import llm_factory
from app.services.intent_tree import IntentTree, IntentNode


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class NodeScore:
    """叶子意图节点打分结果"""
    node_id: str
    node_name: str
    score: float          # 0.0 ~ 1.0，LLM 判断的相关度
    reason: str = ""      # LLM 给出的理由
    # 节点类别，继承自 IntentNode.category，用于下游路由决策：
    #   "operation" → 需二次确认走 RAG 还是执行工具
    #   "concept"   → 直接走 RAG 检索
    category: str = "concept"


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

def _build_classify_prompt(leaf_nodes: list[IntentNode]) -> str:
    """将所有叶子节点构建为分类 prompt"""

    # 构造意图列表供 LLM 选择
    intent_list_lines = []
    for node in leaf_nodes:
        intent_list_lines.append(f'  - id: "{node.id}"')
        intent_list_lines.append(f'    name: "{node.name}"')
        intent_list_lines.append(f'    description: "{node.description}"')
        intent_list_lines.append("")

    intent_list_str = "\n".join(intent_list_lines)

    prompt = f"""你是一个意图分类器。用户会输入一个问题，请你判断该问题与以下每个意图节点的相关程度。

评分规则：
- 1.0：问题明确属于该意图，高置信匹配
- 0.7~0.9：问题与该意图高度相关，可能属于该意图
- 0.4~0.6：问题部分涉及该意图，但主要意图不在此
- 0.1~0.3：仅提及相关概念，不是该意图的核心问题
- 0.0：完全不相关

输出要求：
- 只需输出 JSON 数组，不要解释，不要 Markdown 格式
- 按 score 降序排列，最多返回 10 个意图
- score 为 0 的意图不要出现在结果中

意图列表：
{intent_list_str}

输出格式示例：
[
  {{"id": "op_k8s_deploy", "score": 0.92, "reason": "用户明确询问 K8s 部署流程"}},
  {{"id": "concept_arch", "score": 0.45, "reason": "涉及架构但非核心问题"}}
]

现在请分析用户问题：
"""
    return prompt


def _build_user_question_prompt(question: str) -> str:
    return question


# ---------------------------------------------------------------------------
# 结果解析
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, id2node: dict[str, IntentNode]) -> list[NodeScore]:
    """从 LLM 原始输出解析出 NodeScore 列表"""

    try:
        # 去掉可能的 Markdown 代码块包裹
        cleaned = re.sub(r"^```json\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        raw_data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"[IntentClassifier] JSON 解析失败，尝试正则提取: {e}")
        # 降级：尝试用正则从文本中提取 JSON 片段
        matches = re.findall(r'\{[^{}]*"id"[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
        if not matches:
            logger.error(f"[IntentClassifier] 无法从 LLM 响应中解析出意图结果: {raw[:200]}")
            return []
        try:
            raw_data = json.loads(f"[{','.join(matches)}]")
        except Exception:
            return []

    results: list[NodeScore] = []
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        node_id = item.get("id", "")
        if node_id not in id2node:
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0

        results.append(NodeScore(
            node_id=node_id,
            node_name=id2node[node_id].name,
            score=max(0.0, min(1.0, score)),     # clamp 到 [0, 1]
            reason=str(item.get("reason", "")),
            category=id2node[node_id].category,  # 继承节点的 category
        ))

    return results


# ---------------------------------------------------------------------------
# 分类器接口
# ---------------------------------------------------------------------------

class BaseIntentClassifier:
    """意图分类器基类"""

    def classify(self, question: str) -> list[NodeScore]:
        """对单个问句进行意图分类，返回所有叶子节点的打分（已排序）"""
        raise NotImplementedError


class LLMIntentClassifier(BaseIntentClassifier):
    """LLM 驱动的意图分类器"""

    def __init__(self, intent_tree: IntentTree | None = None):
        self.intent_tree = intent_tree
        self._id2node: dict[str, IntentNode] | None = None
        self._model: Any = None

    def _ensure_init(self):
        if self._id2node is None:
            if self.intent_tree is None:
                from app.services.intent_tree import get_default_intent_tree
                self.intent_tree = get_default_intent_tree()
            self._id2node = self.intent_tree.nodes
            logger.info(f"[IntentClassifier] 初始化完成，共 {len(self.intent_tree.leaf_nodes)} 个叶子意图节点")

    def _get_model(self):
        if self._model is None:
            self._model = llm_factory.create_chat_model(
                model=config.rag_model,
                temperature=0.1,
                streaming=False,
            )
        return self._model

    def classify(self, question: str) -> list[NodeScore]:
        """调用 LLM 对所有叶子意图节点打分"""
        self._ensure_init()
        leaf_nodes = self.intent_tree.leaf_nodes
        if not leaf_nodes:
            logger.warning("[IntentClassifier] 意图树无叶子节点")
            return []

        model = self._get_model()
        system_prompt = _build_classify_prompt(leaf_nodes)

        try:
            response = model.invoke([
                ("system", system_prompt),
                ("human", _build_user_question_prompt(question)),
            ])
            raw_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[IntentClassifier] LLM 调用失败: {e}")
            return []

        scores = _parse_llm_response(raw_text, self._id2node)
        # 按分数降序排列
        scores.sort(key=lambda ns: ns.score, reverse=True)
        logger.debug(f"[IntentClassifier] 分类完成: {question[:30]}... → {len(scores)} 个有效意图")
        return scores


class KeywordIntentClassifier(BaseIntentClassifier):
    """关键词匹配的轻量意图分类器（兜底 / 快速场景）"""

    KEYWORD_MAP: dict[str, list[str]] = {
        "op_k8s_deploy":     ["部署", "安装集群", "创建集群"],
        "op_k8s_scale":      ["扩缩容", "扩容", "缩容", "副本数"],
        "op_k8s_rollback":  ["回滚", "版本回退"],
        "op_k8s_log":        ["日志", "tail", "查看日志"],
        "op_k8s_trouble":    ["故障", "报错", "异常", "CrashLoop", "Pending", "排查"],
        "op_mysql":          ["mysql", "数据库", "主从", "备份", "慢查询"],
        "op_redis":          ["redis", "缓存", "持久化", "内存淘汰"],
        "op_kafka":          ["kafka", "消息队列", "topic", "消费者"],
        "op_nginx":          ["nginx", "反向代理", "负载均衡", "证书"],
        "op_deploy_pipeline":["流水线", "ci/cd", "jenkins", "gitlab ci", "构建"],
        "concept_micro":     ["微服务", "服务治理", "熔断", "限流", "服务网格"],
        "concept_dist":      ["分布式", "一致性", "cap", "base", "paxos", "raft"],
        "concept_cloud":     ["云原生", "容器化", "不可变基础设施"],
        "concept_insurance": ["保险", "理赔", "核保", "保单", "险种"],
        "concept_oa":        ["oa", "考勤", "加班", "请假", "审批"],
        "concept_dev_flow":  ["代码评审", "提测", "发布窗口", "合并规范"],
        "concept_incident":  ["故障处理", "应急响应", "复盘", "定级"],
    }

    def classify(self, question: str) -> list[NodeScore]:
        """基于关键词匹配打分，返回命中的意图节点（按命中次数排序）"""
        q_lower = question.lower()
        scored: dict[str, int] = {}
        for node_id, keywords in self.KEYWORD_MAP.items():
            hit = sum(1 for kw in keywords if kw.lower() in q_lower)
            if hit > 0:
                scored[node_id] = hit

        if not scored:
            return []

        max_hit = max(scored.values())
        results: list[NodeScore] = []
        for node_id, hit_count in scored.items():
            # 归一化分数：命中数越多分数越高，最多命中者给 1.0
            score = round(hit_count / max_hit, 2)
            results.append(NodeScore(
                node_id=node_id,
                node_name=node_id,
                score=score,
                reason=f"关键词命中 {hit_count} 个",
                category="concept",  # 关键词分类器不区分 operation/concept，默认走 RAG
            ))

        results.sort(key=lambda ns: ns.score, reverse=True)
        return results


# ---------------------------------------------------------------------------
# 全局分类器实例（延迟初始化）
# ---------------------------------------------------------------------------

_classifier: BaseIntentClassifier | None = None


def get_intent_classifier(force_keyword: bool = False) -> BaseIntentClassifier:
    """获取全局意图分类器实例

    Args:
        force_keyword: True 时强制使用关键词分类器（无 LLM 调用，用于快速兜底）
    """
    global _classifier
    if _classifier is None:
        if force_keyword:
            _classifier = KeywordIntentClassifier()
            logger.info("[IntentClassifier] 使用关键词分类器（兜底模式）")
        else:
            _classifier = LLMIntentClassifier()
            logger.info("[IntentClassifier] 使用 LLM 分类器")
    return _classifier
