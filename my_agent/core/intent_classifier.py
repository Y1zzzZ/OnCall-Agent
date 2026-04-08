"""
意图识别与路由分诊台 (Intent Classifier & Router)

RAG 系统的第一道大门。用户无论是闲聊、寻求帮助还是查资料，都在这里进行精准分流。
采用极端符合企业级架构的混合方案：
- 第一层 (规则快速过滤)：瞬间拦截纯客套短句，节省大模型 Token 成本。
- 第二层 (模型深度剖析)：针对改写后的长句，调用大模型强校验输出 JSON 取出 `"intent"`。
完全一比一还原你 Java 中的分类器代码思维。
"""
import json
from typing import List, Dict, Any, Tuple
from loguru import logger

from my_agent.core.llm_service import LLMService

class IntentClassifier:
    
    # 纯白话高频词，一旦命中，绝不上报大模型浪费算力
    CHITCHAT_KEYWORDS = {
        "你好", "您好", "谢谢", "感谢", "再见", "拜拜",
        "哈哈", "嗯嗯", "好的", "收到", "明白了", "ok", "在吗", "嗨", "你好呀"
    }
    
    # 允许放行的四种基础门诊科室，杜绝幻觉
    VALID_INTENTS = {"knowledge", "tool", "chitchat", "clarification"}
    
    def __init__(self):
        # 意图分诊理论上只需极其廉价的小模型（比如 7B），这里为了演示仍复用 Qwen 主通道
        self.llm = LLMService()

    async def classify(self, query: str, history_msgs: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        混合分诊方案
        Returns: (intent, confidence)
            intent: knowledge | tool | chitchat | clarification
            confidence: 0.0 ~ 1.0 置信度
        """
        import time
        t0 = time.perf_counter()
        logger.debug(f"[IntentClassifier] 开始分类: '{query}'")

        # 第一层：简单暴力的规则快速拦截网 (短平快)
        if len(query) <= 6 and query.lower() in self.CHITCHAT_KEYWORDS:
            logger.info(f"🚦 [意图分诊] 命中第一层极速规则引擎 (无感白名单): '{query}' => chitchat")
            logger.debug(f"[IntentClassifier] 规则命中耗时: {(time.perf_counter()-t0)*1000:.0f}ms")
            return "chitchat", 0.99
            
        # 第二层：请主治大夫 (大模型) 深度把脉
        try:
            return await self._classify_by_llm(query, history_msgs)
        except Exception as e:
            # 高可用守则：任何报错网络断裂，一律按 "knowledge" 科室兜底扔进去
            logger.warning(f"🚨 [意图分诊] 大模型分诊异常 ({e})，按照灾备方案强制兜底挂号: [knowledge]")
            return "knowledge", 0.5
            
    async def _classify_by_llm(self, query: str, history_msgs: List[Dict[str, str]]) -> Tuple[str, float]:
        history_text = ""
        if not history_msgs:
            history_text = "（无历史对话）"
        else:
            for msg in history_msgs:
                if msg["role"] in ["user", "assistant"]:
                    role_name = "用户" if msg["role"] == "user" else "助手"
                    history_text += f"{role_name}：{msg.get('content', '')}\n"

        prompt = (
            "你是一个意图分类助手。根据对话历史和用户的最新消息，判断用户的意图类别。\n\n"
            "意图类别定义：\n"
            "1. knowledge - 知识检索：用户在询问通用产品信息、退换政策规定、办理指南等。\n"
            "2. tool - 工具调用：用户想查询个人私域数据（订单、余额）、申请年假、或执行某个有副作用的操作。\n"
            "3. chitchat - 闲聊对话：用户在单纯的打招呼、道谢、发脾气等，不涉及具体业务。\n"
            "4. clarification - 引导澄清：用户的问题实在太模糊，缺少关键主语和信息，无法归类。\n\n"
            "判断规则：\n"
            "- 如果用户问题涉及'我的'、'查一下'带有明显私域特指的，通常是 tool\n"
            "- 如果问题是在问普适的公共知识，通常是 knowledge\n"
            "务必且只能以严格的 JSON 格式输出：{\"intent\": \"分类结果\", \"confidence\": 0.9}\n"
            "绝对不要输出 JSON 以外的任何分析过程、语气词！\n\n"
            f"对话历史：\n{history_text}\n\n"
            f"用户最新消息：{query}\n"
        )
        
        # 工具箱锁死（分诊护士没有拿手术刀的权力），防止它擅自行动
        messages = [{"role": "user", "content": prompt}]
        # 冰封 temperature，确保输出最机械化
        import time
        t0 = time.perf_counter()
        logger.debug(f"[IntentClassifier._classify_by_llm] 调用 LLM（非流式）...")
        result = self.llm.evaluate_tools(messages, available_tools=[], temperature=0.1)
        logger.debug(f"[IntentClassifier._classify_by_llm] LLM 返回，耗时: {(time.perf_counter()-t0)*1000:.0f}ms")
        
        if result["type"] != "text":
            raise ValueError("模型脑雾：在不允许调工具的场合触发了异常函数映射")
            
        return self._parse_json_result(result["content"])

    def _parse_json_result(self, raw_content: str) -> Tuple[str, float]:
        # 清理可能附带的 Markdown 代码块残渣 (容错层)
        clean_json = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_json)
            intent = data.get("intent", "knowledge")
            confidence = float(data.get("confidence", 0.8))
            
            # 如果大模型胡乱生造了不存在的科室（例如生造了 complaint 投诉），强行掰回 knowledge
            if intent not in self.VALID_INTENTS:
                logger.warning(f"[意图分诊] 拒绝非法的自造意图狂想 '{intent}'，强制兜底降级回本行: [knowledge]")
                intent = "knowledge"
                confidence = 0.5
                
            return intent, confidence
            
        except json.JSONDecodeError as e:
            logger.warning(f"[意图分诊] 提取报告乱码(非标准 JSON)，无法解析: {e}。强制兜底降级: [knowledge]")
            return "knowledge", 0.5
