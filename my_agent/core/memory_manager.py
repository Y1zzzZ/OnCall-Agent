"""
会话记忆管理器 (Session Memory Manager)

核心策略：基于大模型摘要压缩的极速长期记忆机制 (混合记忆模式)
当早期的瞎哈啦填满阈值（Threshold），残忍压缩为几百字的冷彻骨摘要，但把最新几轮原封不动贴在脸上，以保证对用户的极端敏感。
完美地把你 Java 版的 `SummaryMemory` 和 `calculateChunkBudget` 灵魂转移成了 Python。
"""

"""
    1. 存储位置：数据存储在内存中的两个字典里（store和summary_store）
    2. 存储格式：原始对话以消息列表形式存储，摘要以字符串形式存储
    3. 存储时机：每次对话结束后立即存储
    4. 压缩机制：当对话积累过多时，自动将早期对话压缩为摘要
    5. 读取方式：构建对话历史时同时包含摘要和最近完整对话
"""
from typing import List, Dict, Any, Tuple
from loguru import logger
import json

from my_agent.core.llm_service import LLMService

class SummaryMemoryManager:
    def __init__(self, token_threshold: int = 1500, keep_recent_rounds: int = 3):
        """
        Args:
            token_threshold: 触发大模型摘要压缩的 Token 警戒线 (类似垃圾回收的阈值)
            keep_recent_rounds: 即使压缩，也要雷打不动保留的最近 N 轮完整对话 (1轮=一问一答，包含 2 条)
        """
        self.token_threshold = token_threshold
        self.keep_recent_rounds = keep_recent_rounds
        
        # TODO 暂代生产环境的 Redis 快速主存 后续替换成 redis mysql 高可用
        # 存放形如：sessionId -> [{"role": "user", "content": "我要退货"}]
        self.store: Dict[str, List[Dict[str, str]]] = {}
        # 存放形如：sessionId -> "该用户是一个脾气很差想要退货的人..."
        self.summary_store: Dict[str, str] = {}
        """
          - self.store: 存储原始对话记录，key是会话ID，value是消息列表
          - self.summary_store: 存储压缩后的摘要，key是会话ID，value是摘要文本
        """
        # 挂载自己的云端大脑用来做摘要（不需要工具库，纯陪聊模式）
        self.llm = LLMService()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """极速低内存开销 Token 估算：完全对齐你 Java 单步估计算法"""
        if not text: return 0
        chinese_chars = 0
        other_chars = 0
        for c in text:
            # 判断是否汉字 \u4e00 ~ \u9fff
            if '\u4e00' <= c <= '\u9fff':
                chinese_chars += 1
            elif not c.isspace():
                other_chars += 1
        return int(chinese_chars * 1.5 + other_chars / 4.0)

    def _estimate_total_tokens(self, session_id: str) -> int:
        messages = self.store.get(session_id, [])
        return sum(self.estimate_tokens(msg.get("content", "")) for msg in messages)

    async def add_message(self, session_id: str, role: str, content: str):
        """塞入新的记忆对话，并探针是否击穿阈值，准备进行摘要压榨"""
        if session_id not in self.store:
            self.store[session_id] = []
            
        self.store[session_id].append({"role": role, "content": content})
        
        total_tokens = self._estimate_total_tokens(session_id)
        if total_tokens > self.token_threshold:
            logger.warning(f"🚨 [记忆存储] Token 积压已达警戒线: {total_tokens}/{self.token_threshold}，准备异步清理陈年老旧语料...")
            try:
                await self.compress_early_memory(session_id)
            except Exception as e:
                logger.error(f"摘要压缩异常中断，先放任运行: {e}")

    async def compress_early_memory(self, session_id: str):
        """大模型进行降维打击，早年废话无情压缩"""
        all_messages = self.store.get(session_id, [])
        
        # 一轮对话有 User 和 Assistant 两条，所以乘以 2
        keep_count = self.keep_recent_rounds * 2
        if len(all_messages) <= keep_count:
            return # 压根没那么多话，暂不压缩
            
        # 挥动切割手术刀
        early_messages = all_messages[:-keep_count]
        recent_messages = all_messages[-keep_count:]
        
        conversation_text = ""
        for msg in early_messages:
            conversation_text += f"[{msg['role']}]：{msg['content']}\n"
            
        existing_summary = self.summary_store.get(session_id, "")
        
        # 这是你贴在截图里绝对经过实战淬炼的高级提词
        summary_prompt = (
            "请将以下对话历史压缩为一段简洁的摘要，要求：\n"
            "1. 保留用户的核心意图和关注点\n"
            "2. 保留所有关键实体（产品名、订单号、日期、金额等）\n"
            "3. 保留已经确认的结论和决定\n"
            "4. 保留尚未解决的问题\n"
            "5. 省略寒暄、重复确认、无关细节\n"
            "6. 摘要以第三人称描述，控制在 200 字以内\n"
        )
        if existing_summary:
            summary_prompt += f"\n已有的历史摘要：\n{existing_summary}\n"
        summary_prompt += f"\n需要压缩的新对话记录：\n{conversation_text}"
        
        messages = [
            {"role": "system", "content": "你是一个对话摘要压缩器。"},
            {"role": "user", "content": summary_prompt}
        ]
        
        logger.debug(f"🧹 [记忆清洗] 倒掉 {len(early_messages)} 条陈词滥调，请求千问模型进行神之浓缩...")
        
        # 借用你手搓的核心 LLM 通信器请求运算（不许它在这时调工具！）
        result_box = self.llm.evaluate_tools(messages, available_tools=[])
        if result_box["type"] == "text":
            new_summary = result_box["content"]
            self.summary_store[session_id] = new_summary
            self.store[session_id] = recent_messages # 舍弃老代码，只贴补新的轮次
            logger.success(f"🎊 [记忆清洗] 瘦身成功！历史几十句话坍缩成了: {new_summary[:35]}...")
        else:
            logger.error("大模型生成摘要时不知所云。")

    def build_memory_messages(self, session_id: str) -> Tuple[List[Dict[str, str]], int]:
        """
        给搜索中枢 (Orchestrator) 拼装历史时间线：
        老爷爷的回忆(Summary) + 最近几天的连载(Recent History)
        不仅拼包，还算好它的过路费 (Tokens数量)
        """
        memory_messages = []
        
        # 梯队 1：冷酷的精炼摘要插在最上面
        summary = self.summary_store.get(session_id, "")
        if summary:
            memory_messages.append({
                "role": "system",
                "content": f"【以下是与该用户过往聊天的背景摘要，必须熟记】：\n{summary}"
            })
            
        # 梯队 2：原汁原味的近期实录贴在底下
        recent_messages = self.store.get(session_id, [])
        memory_messages.extend(recent_messages)
        
        history_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in memory_messages)
        return memory_messages, history_tokens

    @staticmethod
    def calculate_chunk_budget(history_tokens: int) -> int:
        """
        最硬核架构设计点：动态调整策略 (Dynamic Budget)
        随着大模型“历史肚子”越来越撑，留给 RAG Chunk 去检索投喂的饭量怎么算？
        对应你 Java 设计图纸上的 `calculateChunkBudget(int historyTokens)`
        """
        total_budget = 8000 # 按照 8K 穷人版模型估命
        system_prompt_tokens = 500  # 固定开销
        reserved_for_generation = 1500 # 还要给回答留 1500 以备大论述
        query_tokens = 100 
        
        available_for_chunks = (
            total_budget - system_prompt_tokens - reserved_for_generation - query_tokens - history_tokens
        )
        
        # 无论日子多穷苦，确保至少能往上下文塞进 1 到 2 个 chunk 破局 (最低 500 口粮)
        return max(500, available_for_chunks)
