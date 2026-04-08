"""
大语言模型服务 (LLM Service)与函数调用核心

主要用于处理与大模型云端 API（如 SiliconFlow）的对话生成以及 Function Call 意图识别。
支持流式与非流式两种调用模式。
"""

import os
import re
import json
from types import SimpleNamespace
from loguru import logger
from typing import List, Dict, Any, Iterator, Union, Optional
# 我们使用兼容 OpenAI 格式的 SDK 来调用各种开源/闭源模型
from openai import OpenAI, Stream

ChunkEvent = Union[Dict[str, Any], str]
"""
evaluate_tools_stream 返回的事件类型：
  - {"type": "tool_calls", "tool_calls": [...]}: 模型决定调用工具
  - {"type": "chunk", "delta": str}: 文本片段（仅纯文本回复时）
"""


def _try_parse_tool_code_plaintext(content: str) -> Optional[List[Any]]:
    """
    部分模型在流式或兼容模式下会把工具调用写成普通文本，例如：
    <tool_code> {"name": "query_internal_knowledge_base", "arguments": {...}} </tool_code>
    解析为可供 Orchestrator 执行的伪 tool_calls 列表。
    """
    if not content or "<tool_code>" not in content.lower():
        return None
    m = re.search(r"<tool_code>\s*(.*?)\s*</tool_code>", content, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("tool_name")
    if not name:
        return None
    args = data.get("arguments")
    if args is None:
        args = data.get("args") or {}
    if isinstance(args, str):
        arg_str = args
    else:
        arg_str = json.dumps(args, ensure_ascii=False)
    fake_call = SimpleNamespace(
        id="parsed-tool-code-1",
        function=SimpleNamespace(name=name, arguments=arg_str),
    )
    return [fake_call]


class LLMService:
    def __init__(self, model_name: str = "qwen-plus"):
        """
        初始化 LLM 客户端
        Args:
            model_name: 必须选择支持 Function Call 的对齐模型
        """
        # 阿里百炼 (DashScope) 兼容 OpenAI 的接口地址
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-f007c0ff34ab4b93882e5434ef25102c")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name

    def generate_answer(self, messages: List[Dict[str, str]]) -> str:
        """主流程 RAG 的终极生成：纯粹根据拼接好的 Prompt 输出最终回答"""
        logger.info(f"正在请求大模型 {self.model_name} 进行最终总结...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content

    def generate_answer_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        流式生成回答：边生成边 yield 每个 token，供 SSE 实时推送。
        适用于最终回复阶段（tool_calls 之后）的纯文本输出流。
        """
        logger.info(f"正在请求大模型 {self.model_name} 进行流式生成...")
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def evaluate_tools_stream(
        self, messages: List[Dict[str, str]], available_tools: List[Dict[str, Any]] = None, temperature: float = 0.1
    ) -> Iterator[ChunkEvent]:
        """
        统一流式策略（参考 Java SSE 客户端逻辑）：

        始终使用 stream=True，逐 chunk 解析 delta：
          - delta.content  → yield chunk，前端实时显示
          - delta.tool_calls（任意一个 chunk 中出现） → 立即 yield tool_calls 并 return
            （停止推文字，转入工具执行分支）

        当 finish_reason == "tool_calls" 时，模型想调用工具，停止 yield 文字。
        当 finish_reason == "stop" 时，模型输出完毕，yield text 结束。
        """
        tools = available_tools or []
        logger.info(f"发送请求（stream=True, tools={len(tools)}）...")

        stream_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            stream_kwargs["tools"] = tools

        stream: Stream = self.client.chat.completions.create(**stream_kwargs)

        accumulated_tool_calls: List[Any] = []   # 跨 chunk 收集 tool_calls
        full_content = ""

        for chunk in stream:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # ── 发现 tool_calls：立即停止推文字，转工具执行分支 ──
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    accumulated_tool_calls.append(tc)
                if finish_reason == "tool_calls":
                    logger.info(f"模型决定调用 {len(accumulated_tool_calls)} 个工具（流式首轮即命中）！")
                    yield {"type": "tool_calls", "tool_calls": accumulated_tool_calls}
                    return

            # ── 文字 token：边生成边 yield，实时推给前端 ──
            if delta.content:
                full_content += delta.content
                yield {"type": "chunk", "delta": delta.content}

            # ── 流结束 ──
            if finish_reason in ("stop", "length"):
                # stop：正常结束，可能有 content
                # length：达到 max_tokens 上限，内容被截断
                pass

        logger.info("流结束，yield text 事件。")
        yield {"type": "text", "content": full_content}

    def evaluate_tools(self, messages: List[Dict[str, str]], available_tools: List[Dict[str, Any]] = None, temperature: float = 0.1) -> Dict[str, Any]:
        """
        处理外部工具调用的函数 (Function Call)，非流式版。
        内部复用 evaluate_tools_stream；必须消费最后的 text 事件，否则 content 恒为空（意图分类会炸）。
        """
        last_text = ""
        for event in self.evaluate_tools_stream(messages, available_tools, temperature):
            et = event["type"]
            if et == "tool_calls":
                return {"type": "tool_calls", "tool_calls": event["tool_calls"]}
            if et == "text":
                last_text = event.get("content") or ""
        return {"type": "text", "content": last_text}
