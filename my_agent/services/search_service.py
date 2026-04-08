"""
中枢大脑调度器 (Orchestrator Search Service)

完美复刻你在 Ragent Java 工程里体会到的宏大设计：
串联 Router (意图工具分配), 执行重入 (Recursive Loop 反思循环), 以及大生态 MCP Client 的挂载。
它是 Python_ragent 真正的灵魂脏器。
"""

import json
import asyncio
from loguru import logger
from typing import List, Dict, Any, AsyncIterator

from my_agent.core.tool_registry import ToolRegistry
from my_agent.core.mcp_client import HttpMCPClient
from my_agent.core.llm_service import LLMService
# 导入我们视若珍宝的主干检索流水线
from my_agent.services.search_pipeline import run_internal_rag_pipeline
# 导入记忆中枢
from my_agent.core.memory_manager import SummaryMemoryManager
# 导入改写中枢
from my_agent.core.query_rewriter import QueryRewriter
# 导入分流守卫
from my_agent.core.intent_classifier import IntentClassifier

class OrchestratorService:
    def __init__(self, mcp_urls: List[str] = None):
        """
        Args:
            mcp_urls: Optional[List[str]] = ["http://localhost:8085"]
        """
        self.llm = LLMService()
        self.registry = ToolRegistry()
        self.mcp_client = HttpMCPClient()
        self.memory = SummaryMemoryManager() # 挂载我们刚刚手搓的记忆神经元
        self.rewriter = QueryRewriter()      # 挂载绝妙的指代消解神镜
        self.classifier = IntentClassifier() # 挂载刚完工的分科室分诊台
        self.mcp_urls = mcp_urls or []
        
    async def initialize(self):
        print(f"DEBUG: Orchestrator.initialize() called. mcp_urls={self.mcp_urls}")
        # 第一发核心子弹：无条件把本地 RAG 作为最高权限金牌库挂载进工具池！
        rag_schema = {
            "name": "query_internal_knowledge_base",
            "description": "这是大模型最核心的武器！当用户询问本公司的业务机密规定、规章制度、内部过往销售规范、文件资料等专属私域数据时，必须调用此工具进行检索。不要尝试用你大脑里的常识编造！",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "极其凝练概括后的搜索关键词（过滤掉聊天的口语）"
                    }
                },
                "required": ["query"]
            }
        }
        self.registry.register_local_tool("query_internal_knowledge_base", rag_schema, run_internal_rag_pipeline)
        
        # 第二发子弹：如果配置了外部微服务，去发 HTTP list 抓生态回来
        if self.mcp_urls:
            await self.registry.initialize_mcp_servers(self.mcp_urls)
            
        logger.info("🧠 天网中枢启运完毕！所有的本地函数流与 MCP 节点群已装填进入待发状态。")

    async def _force_rag_search(self, query: str, collection_name: str = None) -> str:
        """
        强制 RAG 兜底：在模型可能不主动调用工具时，直接替它完成检索。
        """
        from my_agent.services.search_pipeline import run_internal_rag_pipeline
        try:
            result = await run_internal_rag_pipeline(
                query=query,
                top_k=5,
                collection_name=collection_name
            )
            logger.info(f"🏥 [强制 RAG] 检索完毕，结果长度: {len(result)} 字")
            return result
        except Exception as e:
            logger.error(f"🏥 [强制 RAG] 检索失败: {e}")
            return "（知识库检索失败，无法提供相关内容）"

    async def execute_tool(self, tool_name: str, arguments: str, collection_name: str = None) -> str:
        """
        【十字路口转发枢纽】：全宇宙所有的 Tool Intent 都在这分流。
        左转进本地 Python 本地核心栈，右转进外延万里的 MCP 协议栈。
        """
        try:
            # 千问大模型生成的严格 JSON 字符串，我们需要解析
            args_dict = json.loads(arguments) if arguments else {}
        except Exception as e:
            return f"Error: 工具参数大模型胡编失败，不是合法 JSON 格式。报错: {str(e)}"

        if self.registry.is_local(tool_name):
            logger.info(f"⚙️ 引擎调度 [本地 RAG 函数]: {tool_name} | 参数: {args_dict} | 集合: {collection_name}")
            func = self.registry.get_local_function(tool_name)
            try:
                # 如果是 RAG 工具，尝试注入 collection_name
                if tool_name == "query_internal_knowledge_base" and collection_name:
                    args_dict["collection_name"] = collection_name
                
                # 我们的管道是全双工 async 的，直接等待跑完 RAG
                result = await func(**args_dict) 
                return str(result)
            except Exception as e:
                 logger.error(f"本地 RAG 函数执行严重雪崩: {e}")
                 return f"Error: 内部出错 {str(e)}"
                 
        elif self.registry.is_mcp(tool_name):
            logger.info(f"🌐 引擎调度 [远端 MCP 节点转发]: {tool_name} | 参数: {args_dict}")
            server_url = self.registry.get_mcp_server_url(tool_name)
            # 通过 HTTP (SSE/Rest) 完全不用自己写的协议层，全权托管
            return await self.mcp_client.call_tool(server_url, tool_name, args_dict)
            
        else:
            return f"Error: 工具 {tool_name} 没在注册表中挂号，大模型请重新审视你可用的工具列表！"

    async def chat(self, user_query: str, session_id: str = "default_session", collection_name: str = None) -> str:
        """
        🔥 中枢主干引擎 (Recursive ReAct Loop) 🔥
        负责和千问模型生死共浴的大一统循环。加入记忆混合模式护航。
        """
        logger.info(f"\n======== 新一轮意图博弈启幕 === [Query: {user_query}] === [KB: {collection_name}] ========")
        
        # 第一步：老爷爷讲故事。把以前的记忆摘要以及最近几次鲜活的历史掏出来！
        memory_msgs, history_tokens = self.memory.build_memory_messages(session_id)
        
        # 第二步：中枢调度。看看老家伙占了多大的口粮，精算留给 RAG 碎片的弹药配给
        chunk_budget = self.memory.calculate_chunk_budget(history_tokens)
        logger.debug(f"[运筹帷幄] 历史记忆吃掉了 {history_tokens} 令牌。给前线 RAG (候选碎片池) 拨付 {chunk_budget} 令牌余额！")
        
        # 标准系统指令
        system_prompt = {
            "role": "system",
            "content": "你是由最高统帅打造的专业公司全能助手，如果找不到答案请诚恳拒绝回答。你脑中附带了上下文和知识记忆。"
        }
        
        # 绝妙插入点：通过小模型做一重过滤清洗，把诸如“那它的保修期呢”扩写成“iPhone 16的...”
        # 防止进洞后因为代词太多导致向量全盘迷失。
        actual_query = await self.rewriter.safe_rewrite(user_query, memory_msgs)
        
        # 【中层防波堤：分诊意图路由器】根据刚才改写好的脱水句子，判定该扔到哪个处理管道
        # 为什么把这步放在改写后？尊重你在截图末尾的生产经验备注：绝大部分生产查询依然是RAG，
        # 所以先抹平语意能极大提高 Intent 的识别准度！
        intent, confidence = await self.classifier.classify(actual_query, memory_msgs)
        logger.info(f"🏥 [分诊导流] 正式确诊当前用户意图: 【{intent.upper()}】 (医生自信度: {confidence:.2f})")
        
        # --- 路由第一支线：闲聊区 (Chitchat) ---
        if intent == "chitchat":
            logger.info("👉 命中闲聊快捷通道！这回合算力省下来了，坚决不允许它乱调工具！")
            chitchat_ans = self.llm.evaluate_tools([{"role": "user", "content": actual_query}], available_tools=[])["content"]
            await self.memory.add_message(session_id, "user", user_query)
            await self.memory.add_message(session_id, "assistant", chitchat_ans)
            return chitchat_ans
            
        # --- 路由第二支线：模糊诱导 (Clarification) ---
        if intent == "clarification":
            logger.info("👉 命中诱导拦截通道。防患于未然，没必要瞎搜。")
            clarify_ans = "[反问引导] 主人，您这问题没头没尾真是难住我了😅。能否提供一下准确的产品标识或订单口令？不然我翻遍知识库也不知道搜啥呀..."
            await self.memory.add_message(session_id, "user", user_query)
            await self.memory.add_message(session_id, "assistant", clarify_ans)
            return clarify_ans
            
        # --- 路由终末：正经干活 (Knowledge / Tool) ---
        # 如果走到这，说明要么是要搜 RAG (knowledge)，要么是要查接口 (tool)，这就轮到下方的
        # 递归 ReAct 神器——也就是咱们的 Function Calling + MCP 联动来一锤定音了。
        logger.info("👉 确认为正经业务级请求 (Knowledge 或 Tool)，准予放行，移交核心循环处理槽！")

        # 第三步：发车包裹=系统高压法则 + (历史背景摘要+最近闲聊) + 补全残肢的干净问题
        messages = [system_prompt] + memory_msgs + [{"role": "user", "content": actual_query}]
        
        # 将所有的武器图纸拿出
        available_tools = self.registry.get_all_tools_for_llm()
        
        max_loop = 5 # 绝对熔断阈值，防止小模型变成智障陷入无限疯狂调用的旋涡中
        
        for current_loop in range(max_loop):
            logger.info(f"--- 💡 进入大模型第 {current_loop + 1} 轮深度思考栈 ---")
            
            try:
                # 把当前的上下文全量历史和所有的工具扔进熔炉
                # 这个 evaluate_tools 就是我们在 llm_service 里刚刚新鲜手搓的那个
                result_box = self.llm.evaluate_tools(messages, available_tools)
            except Exception as e:
                logger.error(f"模型大脑云端崩溃或欠费: {str(e)}")
                return "抱歉，目前我的算力大脑处于宕机状态，请稍微喝杯咖啡再试。"
                
            # -------------------------------------------------------------
            # 场景 A：脑波平稳，大模型觉得凭手上的数据它已经完全懂了，直接给出自然语言答案。
            # -------------------------------------------------------------
            if result_box["type"] == "text":
                final_answer = result_box["content"]
                logger.success("🎯 思维链完整成环！大模型给出了终极回答。")
                
                # 双杀式触发入库动作：既存档用户的质问，也存档模型的辩护 (触发自动摘要压缩判定)
                await self.memory.add_message(session_id, "user", user_query)
                await self.memory.add_message(session_id, "assistant", final_answer)
                
                return final_answer
                
            # -------------------------------------------------------------
            # 场景 B：出现变数！大模型发出了 Tool Call 指令意图！
            # -------------------------------------------------------------
            elif result_box["type"] == "tool_calls":
                # 也就是大一统 JSON 对象
                tool_calls = result_box["tool_calls"]
                
                # 【巨核铁律】必须把大模型的这条『我想调用什么』的思考痕迹塞进历史 (且 content必须为 None)
                # 不然下轮模型会因为 OpenAI Context Validation 崩坏而直接 400 报错。
                # 这是一般新手极其容易踩进去的大坑！
                messages.append({
                    "role": "assistant",
                    "content": None,  
                    "tool_calls": tool_calls
                })
                
                # 真正的脏活苦活轮到咱们写的中枢来干了
                for tool_call in tool_calls:
                    tool_id = tool_call.id
                    func_name = tool_call.function.name
                    func_args = tool_call.function.arguments
                    
                    # 强悍的代理转发，管你是去 Milvus 还是去 2000公里外某个 Java MCP Server 呢，反正都在这
                    tool_result_str = await self.execute_tool(func_name, func_args, collection_name=collection_name)
                    
                    # 规制：必须把答案包装成 role="tool" ，配上对应该题的 tool_id，交还给大模型判卷！
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": func_name,
                        "content": tool_result_str
                    })
                    
        fallback_msg = "抱歉，由于操作步骤过于繁琐且耗去了我过度算力（已达 5 次递归上限），系统停止思考并退出了执行栈。"
        await self.memory.add_message(session_id, "user", user_query)
        await self.memory.add_message(session_id, "assistant", fallback_msg)
        return fallback_msg

    async def chat_stream(self, user_query: str, session_id: str = "default_session", collection_name: str = None) -> AsyncIterator[str]:
        """
        🔥 流式版中枢主干引擎 (Recursive ReAct Loop) 🔥
        边生成边 yield token，前端可实时收到推送。
        工具调用阶段仍为同步批量执行，仅最终文本回复阶段流式输出。
        """
        import time
        t_start = time.perf_counter()
        t_last = t_start

        def _log_ms(step: str):
            nonlocal t_last
            now = time.perf_counter()
            elapsed = (now - t_last) * 1000
            total = (now - t_start) * 1000
            logger.info(f"  ⏱ [{step}] +{elapsed:.0f}ms  (累计 +{total:.0f}ms)")
            t_last = now

        logger.info(f"\n======== 流式对话启幕 === [Query: {user_query}] === [KB: {collection_name}] ========")

        memory_msgs, history_tokens = self.memory.build_memory_messages(session_id)
        chunk_budget = self.memory.calculate_chunk_budget(history_tokens)
        _log_ms("① Memory 构建完毕")

        system_prompt = {
            "role": "system",
            "content": "你是由最高统帅打造的专业公司全能助手，如果找不到答案请诚恳拒绝回答。你脑中附带了上下文和知识记忆。"
        }

        # Query改写 + 意图分类 并行执行，节省 2~3 秒
        rewrite_task = self.rewriter.safe_rewrite(user_query, memory_msgs)
        classify_task = self.classifier.classify(user_query, [])  # 意图分类无需历史上下文，并行更纯粹
        actual_query, (intent, confidence) = await asyncio.gather(rewrite_task, classify_task)
        _log_ms("② Query改写+③ 意图分类 并行完毕")
        logger.info(f"🏥 [分诊导流] 意图: 【{intent.upper()}】 (自信度: {confidence:.2f})")

        if intent == "chitchat":
            logger.info("👉 命中闲聊快捷通道，流式回复")
            for token in self.llm.generate_answer_stream([{"role": "user", "content": actual_query}]):
                yield token
                await asyncio.sleep(0)
            await self.memory.add_message(session_id, "user", user_query)
            return

        if intent == "clarification":
            logger.info("👉 命中诱导拦截通道，流式回复")
            clarify_ans = "[反问引导] 主人，您这问题没头没尾真是难住我了😅。能否提供一下准确的产品标识或订单口令？不然我翻遍知识库也不知道搜啥呀..."
            for token in clarify_ans:
                yield token
                await asyncio.sleep(0)
            await self.memory.add_message(session_id, "user", user_query)
            return

        logger.info("👉 确认为正经业务级请求 (Knowledge 或 Tool)，移交核心循环处理槽！")

        # ── 【强制 RAG 兜底】KNOWLEDGE 意图：模型大概率不主动调工具，直接替它搜 ──
        if intent == "knowledge":
            logger.info("🏥 [强制 RAG] KNOWLEDGE 意图，绕过模型犹豫，直接检索知识库！")
            rag_result = await self._force_rag_search(actual_query, collection_name)
            _log_ms("④ RAG 检索完毕")
            forced_rag_msg = (
                f"【以下是从公司内部知识库检索到的相关材料，请严格依据这些材料作答，禁止编造】\n"
                f"{rag_result}\n\n"
                f"用户问题：{actual_query}"
            )
            # 直接把 RAG 结果注入到首条 user 消息，模型只能基于这些材料回答
            messages = [
                system_prompt,
                {"role": "user", "content": forced_rag_msg}
            ]
        else:
            messages = [system_prompt] + memory_msgs + [{"role": "user", "content": actual_query}]

        # evaluate_tools_stream 始终 stream=True，带 tools 时收到 tool_calls 立即返回，
        # 否则 content 边生成边 yield，实现真正流式推送。
        available_tools = self.registry.get_all_tools_for_llm()

        max_loop = 5

        for current_loop in range(max_loop):
            logger.info(f"--- 💡 进入大模型第 {current_loop + 1} 轮深度思考栈 ---")

            full_answer = ""
            tool_calls_found = None
            try:
                t_llm_start = time.perf_counter()
                first_token_received = False
                for event in self.llm.evaluate_tools_stream(messages, available_tools):
                    etype = event["type"]
                    if etype == "tool_calls":
                        tool_calls_found = event["tool_calls"]
                        _log_ms(f"⑥ LLM 流式结束 (tool_calls 命中)")
                        break
                    elif etype == "chunk":
                        if not first_token_received:
                            first_ms = (time.perf_counter() - t_llm_start) * 1000
                            logger.info(f"  ⏱ [⑤ LLM 首个 token] +{first_ms:.0f}ms (累计 +{(time.perf_counter()-t_start)*1000:.0f}ms)")
                            first_token_received = True
                        token = event["delta"]
                        full_answer += token
                        yield token
                        await asyncio.sleep(0)
                    elif etype == "text":
                        _log_ms(f"⑥ LLM 流式结束 (输出完毕，共 {len(full_answer)} 字)")
            except Exception as e:
                logger.error(f"模型大脑云端崩溃或欠费: {str(e)}")
                fallback = "抱歉，目前我的算力大脑处于宕机状态，请稍微喝杯咖啡再试。"
                for token in fallback:
                    yield token
                return

            if tool_calls_found:
                tool_calls = tool_calls_found
                logger.info(f"模型决定调用 {len(tool_calls)} 个工具，进入工具执行分支。")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })

                for tool_call in tool_calls:
                    tool_id = tool_call.id
                    func_name = tool_call.function.name
                    func_args = tool_call.function.arguments

                    tool_result_str = await self.execute_tool(func_name, func_args, collection_name=collection_name)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": func_name,
                        "content": tool_result_str
                    })
                continue

            # 流结束，无 tool_calls → 纯文本回复，已通过 chunk 事件逐 token yield 完毕
            if full_answer:
                logger.success("🎯 思维链完整成环！大模型给出了终极回答，流式推送已完成。")
                await self.memory.add_message(session_id, "user", user_query)
                await self.memory.add_message(session_id, "assistant", full_answer)
            return

        fallback_msg = "抱歉，由于操作步骤过于繁琐且耗去了我过度算力（已达 5 次递归上限），系统停止思考并退出了执行栈。"
        await self.memory.add_message(session_id, "user", user_query)
        await self.memory.add_message(session_id, "assistant", fallback_msg)
        for token in fallback_msg:
            yield token
