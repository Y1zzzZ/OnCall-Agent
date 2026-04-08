"""
RAG 系统提示词管理 (Prompts)

负责管理大语言模型的“系统指令 (System Prompt)”。
工业界最佳实践是将这类字符串写在独立的 Markdown/YAML，或外部的 Prompt 管理后台 (如 Dify) 中，
以实现代码与业务逻辑的分离。为了 MVP 演示，我们先以常量模板形式固化在代码里。
"""

# 这是专门为 RAG 场景打磨的中文系统提示词
# 核心解决这三个问题：幻觉、乱答、无溯源
RAG_SYSTEM_PROMPT = """你是一个严谨且专业的企业级智能知识助手。
你的核心任务是：**严格根据给定的【参考上下文】**来回答用户的问题。

在回答时，你必须遵守以下三大铁律：
1. **绝对禁止幻觉 (No Hallucination)**：你所有的回答必须从上下文中推理得出。如果上下文中并未提及用户提问的内容，或者上下文的信息不足以得出结论，请直接回答：“抱歉，目前的知识库中没有检索到相关的参考信息。” 绝对**不允许**动用你自身的预训练知识进行瞎编！
2. **直击痛点 (Relevance)**：用户的提问是什么，就精简干练地回答什么，不要做多余的发散。
3. **强制溯源 (Traceability)**：在引用具体的数据、规则或原文描述时，请在句子末尾附上来源标记（比如：引用自 [文件名]）。

---
以下是系统为你自动检索出的【参考上下文】：
{context}
"""

def build_rag_messages(context_chunks: list, user_query: str) -> list:
    """
    核心拼接器：把检索结果、提示词铁律、用户问题 融合成大模型认识的格式。
    
    Args:
        context_chunks: 之前从 Milvus 和 Rerank 里历经千难万险拿出的高分文本列表。
        user_query: 用户真正的原始提问（例如："公司今年的营收是多少"）
    """
    
    # 1. 组装上下文流
    # 这里我们把多个 chunk 用分隔符拼起来，如果有 source 就标上 source
    context_str_list = []
    for idx, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        # 如果前面我们存了 metadata，这里可以把它提出来用于溯源
        # 假设这里有 source
        source = chunk.get("source", f"文档块{idx+1}")
        context_str_list.append(f"【信息片段 {idx+1}】(来源: {source}):\n{text}")
        
    final_context_str = "\n\n".join(context_str_list)
    
    # 2. 把最终的干货文本，通过字符串替换 (format) 塞进我们的宏大铁律模板中
    system_content = RAG_SYSTEM_PROMPT.format(context=final_context_str)
    
    # 3. 按 OpenAI / Qwen 统一的标准 Chat 格式组装
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query}
    ]
    
    return messages

