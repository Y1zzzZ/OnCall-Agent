"""
Query 改写器 (Query Rewriter)

基于指代消解 (Coreference Resolution) 和上下文补全机制，
确保最终发送给向量库大本营的检索词是脱离聊天的、完全独立的完整事实断句。
例如：“那它的保修期呢？” -> “iPhone 16 Pro 的保修期是什么？”
完全完美对齐 Java 的企业级重构与兜底设计。
"""
import re
from typing import List, Dict, Any
from loguru import logger

from my_agent.core.llm_service import LLMService

class QueryRewriter:
    def __init__(self):
        self.llm = LLMService()
        
    def needs_rewrite(self, query: str, history_msgs: List[Dict[str, str]]) -> bool:
        """
        [成本与时机控制] 
        基于极其轻量级的规则判断，拦截掉不需要花费大模型 Token 被迫改写的长难句。
        完美对齐你提供的 Java `needsRewrite` 拦截器逻辑。
        """
        # 第一轮对话且 query 足够长，大概率不需要补全
        if not history_msgs and len(query) > 15:
            return False
            
        # 包含代词，必须改写 (指代消解的心脏规则)
        if re.search(r"[它这那][个些]?|上面|刚才", query):
            return True
            
        # query 太短，大概率省略了上下文主语 (比如：“价格呢？”)
        if len(query) < 10 and history_msgs:
            return True
            
        # 只要有历史对话，安全起见默认开启改写防丢失上下文
        return bool(history_msgs)

    async def _rewrite_core(self, query: str, history_msgs: List[Dict[str, str]]) -> str:
        """调用 LLM 进行深度代词指代消解和语句结构主谓宾补全"""
        history_text = ""
        if not history_msgs:
            history_text = "（无历史对话）"
        else:
            for msg in history_msgs:
                # 只提纯 user 和 assistant 的干货
                if msg["role"] in ["user", "assistant"]:
                    role_name = "用户" if msg["role"] == "user" else "助手"
                    history_text += f"{role_name}：{msg.get('content', '')}\n"

        # 原汁原味复现你精心调教的极品 Prompt 5条圣旨
        rewrite_prompt = (
            "你是一个查询改写助手。根据对话历史和用户的最新问题，将问题改写为一个独立的、完整的检索查询。\n\n"
            "要求：\n"
            "1. 如果最新问题中包含代词（它、这个、那个等）或省略了关键信息，请结合对话历史补全。\n"
            "2. 如果问题已经足够完整清晰，请原样输出，不要画蛇添足。\n"
            "3. 只输出改写后的查询，不要输出任何解释、前缀或多余内容。\n"
            "4. 改写后的查询应该是一个独立的句子，脱离对话历史也能理解。\n\n"
            f"对话历史：\n{history_text}\n\n"
            f"用户最新问题：{query}\n\n"
            "改写后的查询："
        )
        
        # 组装请求，不要任何工具
        messages = [{"role": "user", "content": rewrite_prompt}]
        
        # [控制幻觉死穴] 为了极致的速度和不可发散性，强行将 temperature 在底层调用处拉低到 0.1
        # （这里暂时用默认 llm_service 给的参数模拟，后续 llm_service 可自行放开温度支持）
        import time
        t0 = time.perf_counter()
        logger.debug(f"[QueryRewriter._rewrite_core] 调用 LLM（非流式）...")
        result = self.llm.evaluate_tools(messages, available_tools=[])
        logger.debug(f"[QueryRewriter._rewrite_core] LLM 返回，耗时: {(time.perf_counter()-t0)*1000:.0f}ms")
        if result["type"] == "text":
            return result["content"].strip()
        else:
            raise RuntimeError("大模型未能正常返回纯文本而是尝试了工具调用")

    async def safe_rewrite(self, query: str, history_msgs: List[Dict[str, str]]) -> str:
        """
        安全改写代理：强韧的报错容错兜底防护网。
        因为任何改写大模型都有可能网络抖动超时崩盘，这时候不能让整个 RAG 宕机！必须用原话查！
        绝对对齐你 Java UI 图纸上的 `safeRewrite` 方法！
        """
        if not self.needs_rewrite(query, history_msgs):
            logger.debug(f"[Query 拦截] 这句话十分纯正脱水，无需耗费算力改写，直接放行: '{query}'")
            return query
            
        logger.info(f"[Query 改造] 抓到模糊代词或口语残句，送入改造工厂: '{query}'")
        try:
            rewritten = await self._rewrite_core(query, history_msgs)
            
            # 极限兜底校验：不能空，不能被大模型写成上百字的小说扯淡文章
            if rewritten and len(rewritten) < 500:
                logger.success(f"✨ [Query 改造] 重塑金身！ '{query}' -> '{rewritten}'")
                return rewritten
            else:
                logger.warning(f"[Query 改造] 大模型脑神经搭错导致改写异常或过长，退回原问题保命！结果: '{rewritten[:50]}'")
                return query
                
        except Exception as e:
            logger.warning(f"🚨 [Query 改造] 模型崩溃或网络超时: {e}，触发最高安全预案：退回原问题兜底保命！")
            return query
