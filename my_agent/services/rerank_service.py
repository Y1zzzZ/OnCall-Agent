"""
重排序服务 (Rerank Service - Cross-Encoder)

利用阿里云百炼 Reranker API，对粗排检索出的文档进行二次打分。

【为什么要重新打分？】
1. 粗检索 (Bi-Encoder) 虽然快（100万文本秒级响应），但 Query 和 Chunk 是独立编码的，无法捕获词汇间的深层交互。
2. 重排序 (Cross-Encoder) 把 Query 和 Chunk 拼接在一起送入模型，精度极高，能深刻理解上下文。
3. 工业标准玩法：快扫 (Bi-Encoder 召回 20-50 个) + 慢审 (Cross-Encoder 精排选出 Top 5)。

【支持的模型（阿里云百炼官方文档 2026-03）】
- qwen3-rerank：最多 500 文档，0.0005元/千Token，支持 100+ 语种
- gte-rerank-v2：最多 30000 文档，0.0008元/千Token，支持 50+ 语种
"""

import os
import requests
from typing import List, Dict, Any
from loguru import logger


class DashScopeRerankService:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "qwen3-rerank"
    ):
        self.api_key = api_key or os.environ.get(
            "DASHSCOPE_API_KEY",
            "sk-f007c0ff34ab4b93882e5434ef25102c"
        )
        self.model_name = model_name

        # qwen3-rerank / gte-rerank-v2 均走此兼容端点，结构稍有不同，下面统一封装
        self.base_url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

        logger.info(f"DashScope 精排服务初始化完成: 使用模型={model_name}，端点={self.base_url}")

    def rerank_documents(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        调用阿里云百炼 Cross-Encoder 进行精准重排序。

        Args:
            query: 用户提问
            documents: 粗排召回的内容列表（建议 20~50 个）
            top_n: 精排后最终保留几个

        Returns:
            List[Dict]: 按相关性得分从高到低排序的结果
            形如: [{"index": 0, "text": "...", "score": 0.95}, ...]
        """
        if not documents:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # qwen3-rerank 使用 compatible-api/v1/reranks，请求体参数平铺
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
            # 问答检索（默认），也可切 "Retrieve semantically similar text." 做语义相似
            "instruct": "Given a web search query, retrieve relevant passages that answer the query."
        }

        logger.info(f"正在使用 {self.model_name} 对 {len(documents)} 个候选块进行交叉精排...")

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # 兼容两种响应结构：
            # 1. compatible-api 格式：{"results": [...]}
            # 2. 标准 v1 格式：{"output": {"results": [...]}}
            raw_results = (
                data.get("results")
                or data.get("output", {}).get("results")
                or []
            )

            if not raw_results:
                logger.warning(f"精排返回结果为空，降级返回粗排前 {top_n} 条。")
                return [{"index": i, "text": doc, "score": 0.0} for i, doc in enumerate(documents[:top_n])]

            logger.info(f"精排完成。最高分: {raw_results[0].get('relevance_score', 0):.4f}")

            formatted_results = []
            for item in raw_results:
                # compatible-api 返回: {"index": 0, "relevance_score": 0.9, "document": {"text": "..."}}
                # 标准 v1 返回: same structure
                doc_obj = item.get("document", {})
                doc_text = doc_obj.get("text", "") if isinstance(doc_obj, dict) else str(doc_obj)
                formatted_results.append({
                    "index": item.get("index"),
                    "text": doc_text,
                    "score": item.get("relevance_score"),
                })

            return formatted_results

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body = e.response.text if e.response is not None else ""
            logger.error(f"Reranker HTTP 错误 {status}：{body}")
        except Exception as e:
            logger.error(f"Reranker 重排序调用失败: {e}")

        logger.warning("精排失败，降级返回粗排的原始前 Top-N 个结果！")
        return [{"index": i, "text": doc, "score": 0.0} for i, doc in enumerate(documents[:top_n])]
