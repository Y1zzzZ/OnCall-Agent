"""意图树数据模型

意图树是一棵多叉树，叶子节点对应具体可检索的意图（如 "保险理赔流程"、"OA考勤规则"）。
内部节点用于分类组织（如 "保险系统 > 理赔"，"业务系统 > OA系统"）。

数据示例（可替换为从配置文件或数据库加载）:
    保险系统
        ├── 理赔
        │   ├── 理赔流程          ← 叶子节点
        │   ├── 理赔材料          ← 叶子节点
        │   └── 理赔时效          ← 叶子节点
        └── 产品
            ├── 产品介绍          ← 叶子节点
            └── 产品费率          ← 叶子节点
    业务系统
        └── OA系统
            ├── 考勤规则          ← 叶子节点
            └── 加班申请          ← 叶子节点
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentNode:
    """意图树节点"""

    id: str = field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:8]}")
    name: str = ""                                  # 节点显示名称，如 "理赔流程"
    description: str = ""                           # 节点描述，供 LLM 理解该意图含义
    parent_id: Optional[str] = None                 # 父节点 ID
    children: list[IntentNode] = field(default_factory=list)
    is_leaf: bool = False                          # 是否为叶子节点（可检索）
    # 节点类别，决定下游路由：
    #   "operation" → 可能是操作意图（需要二次确认走 RAG 还是执行工具）
    #   "concept"   → 纯知识检索，直接走 RAG
    category: str = "concept"                       # 默认走 RAG（保守兜底）

    def __post_init__(self):
        self.is_leaf = len(self.children) == 0


@dataclass
class IntentTree:
    """意图树，包含所有节点和叶子节点集合"""

    nodes: dict[str, IntentNode] = field(default_factory=dict)
    root_ids: list[str] = field(default_factory=list)    # 可能有多个根（多业务线）

    @property
    def leaf_nodes(self) -> list[IntentNode]:
        """返回所有叶子节点"""
        return [n for n in self.nodes.values() if n.is_leaf]

    def get_node(self, node_id: str) -> Optional[IntentNode]:
        return self.nodes.get(node_id)

    def get_leaf_by_id(self, node_id: str) -> Optional[IntentNode]:
        node = self.get_node(node_id)
        return node if node and node.is_leaf else None


# ---------------------------------------------------------------------------
# 预置意图树（企业内务/IT运维场景，可按需扩展）
# ---------------------------------------------------------------------------

def build_default_intent_tree() -> IntentTree:
    """构建默认意图树

    意图分为两大类：
    - operation：操作类（部署/启停/配置/故障排查），category="operation"
    - concept：概念类（原理/流程/规则/定义），category="concept"

    每个叶子节点包含 id、name、description、category，便于 LLM 理解并打分，
    便于下游路由决策。
    """

    # ---------- operation 类 ----------
    operation_root = IntentNode(
        id="operation",
        name="操作运维",
        description="操作运维类问题，包括部署、启停、配置、扩缩容、故障排查等",
    )

    # K8s 相关
    k8s_node = IntentNode(id="operation_k8s", name="Kubernetes", parent_id="operation")
    k8s_deploy = IntentNode(id="op_k8s_deploy", name="K8s部署", parent_id="operation_k8s",
                            description="Kubernetes 集群部署、副本管理、节点扩缩", category="operation")
    k8s_scale = IntentNode(id="op_k8s_scale", name="K8s扩缩容", parent_id="operation_k8s",
                           description="Pod 扩缩容、HPA 策略、扩容阈值配置", category="operation")
    k8s_rollback = IntentNode(id="op_k8s_rollback", name="K8s回滚", parent_id="operation_k8s",
                              description="Deployment 回滚、版本切换、历史版本查看", category="operation")
    k8s_log = IntentNode(id="op_k8s_log", name="K8s日志查询", parent_id="operation_k8s",
                         description="Pod 日志查看、容器日志、实时 tail", category="operation")
    k8s_troubleshoot = IntentNode(id="op_k8s_trouble", name="K8s故障排查", parent_id="operation_k8s",
                                  description="Pod CrashLoopBackOff、Pending、ImagePullBackOff 等故障诊断", category="operation")

    # 数据库相关
    db_node = IntentNode(id="operation_db", name="数据库", parent_id="operation")
    db_mysql = IntentNode(id="op_mysql", name="MySQL运维", parent_id="operation_db",
                          description="MySQL 启停、主从复制、慢查询优化、备份恢复", category="operation")
    db_redis = IntentNode(id="op_redis", name="Redis运维", parent_id="operation_db",
                          description="Redis 集群、缓存策略、持久化、内存淘汰", category="operation")
    db_mongo = IntentNode(id="op_mongodb", name="MongoDB运维", parent_id="operation_db",
                          description="MongoDB 副本集、分片集群、索引管理", category="operation")

    # 中间件相关
    mw_node = IntentNode(id="operation_middleware", name="中间件", parent_id="operation")
    mw_kafka = IntentNode(id="op_kafka", name="Kafka运维", parent_id="operation_middleware",
                          description="Kafka 集群、Topic 管理、消费者组、消息积压处理", category="operation")
    mw_nginx = IntentNode(id="op_nginx", name="Nginx运维", parent_id="operation_middleware",
                           description="Nginx 配置、热加载、负载均衡、证书管理", category="operation")

    # 部署发布
    deploy_node = IntentNode(id="operation_deploy", name="部署发布", parent_id="operation")
    deploy_pipeline = IntentNode(id="op_deploy_pipeline", name="CI/CD流水线", parent_id="operation_deploy",
                                 description="Jenkins/GitLab CI 流水线、阶段配置、构建日志", category="operation")
    deploy_rollback = IntentNode(id="op_deploy_rollback", name="版本回滚", parent_id="operation_deploy",
                                 description="生产环境版本回滚操作步骤、注意事项", category="operation")
    deploy_config = IntentNode(id="op_deploy_config", name="配置变更", parent_id="operation_deploy",
                               description="应用配置变更、配置文件更新、热更新", category="operation")

    # ---------- concept 类 ----------
    concept_root = IntentNode(
        id="concept",
        name="概念原理",
        description="概念原理类问题，包括架构设计、工作原理、流程规则等",
    )

    # 架构设计
    arch_node = IntentNode(id="concept_arch", name="架构设计", parent_id="concept")
    arch_micro = IntentNode(id="concept_micro", name="微服务架构", parent_id="concept_arch",
                            description="微服务拆分原则、服务治理、熔断限流、服务网格", category="concept")
    arch_dist = IntentNode(id="concept_dist", name="分布式系统", parent_id="concept_arch",
                           description="分布式一致性、CAP/BASE、Paxos/Raft、分布式事务", category="concept")
    arch_cloud = IntentNode(id="concept_cloud", name="云原生概念", parent_id="concept_arch",
                            description="容器化、声明式 API、不可变基础设施、云原生设计理念", category="concept")

    # 业务规则
    rule_node = IntentNode(id="concept_rule", name="业务规则", parent_id="concept")
    rule_insurance = IntentNode(id="concept_insurance", name="保险业务规则", parent_id="concept_rule",
                                 description="保险产品规则、理赔条款、核保规则、保单效力", category="concept")
    rule_oa = IntentNode(id="concept_oa", name="OA办公规则", parent_id="concept_rule",
                         description="OA系统考勤规则、审批流程、加班调休、请假销假", category="concept")

    # 流程定义
    flow_node = IntentNode(id="concept_flow", name="流程定义", parent_id="concept")
    flow_dev = IntentNode(id="concept_dev_flow", name="开发流程", parent_id="concept_flow",
                           description="代码评审流程、合并规范、提测标准、发布窗口", category="concept")
    flow_incident = IntentNode(id="concept_incident", name="故障处理流程", parent_id="concept_flow",
                               description="故障定级、应急响应、复盘流程、升级机制", category="concept")

    # ---------- 汇总 ----------
    all_nodes: dict[str, IntentNode] = {}
    for node in [
        operation_root, k8s_node, k8s_deploy, k8s_scale, k8s_rollback, k8s_log, k8s_troubleshoot,
        db_node, db_mysql, db_redis, db_mongo,
        mw_node, mw_kafka, mw_nginx,
        deploy_node, deploy_pipeline, deploy_rollback, deploy_config,
        concept_root, arch_node, arch_micro, arch_dist, arch_cloud,
        rule_node, rule_insurance, rule_oa,
        flow_node, flow_dev, flow_incident,
    ]:
        all_nodes[node.id] = node

    return IntentTree(nodes=all_nodes, root_ids=["operation", "concept"])


# 全局意图树实例
_default_intent_tree: IntentTree | None = None


def get_default_intent_tree() -> IntentTree:
    global _default_intent_tree
    if _default_intent_tree is None:
        _default_intent_tree = build_default_intent_tree()
    return _default_intent_tree
