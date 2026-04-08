"""
向量化嵌入服务 (Embedding Service)

基于 httpx.AsyncClient 实现全异步 HTTP 请求，不阻塞事件循环。
支持批量处理、多线程并发控制（用于入库场景）、指数退避重试。
"""
import os
import time
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from loguru import logger


class DashScopeEmbeddingService:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "text-embedding-v3",
        batch_size: int = 5,       # 降低批次，减少单次请求体积
        max_threads: int = 3,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.environ.get(
            "DASHSCOPE_API_KEY",
            "sk-f007c0ff34ab4b93882e5434ef25102c"
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_threads = max_threads
        self.max_retries = max_retries
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        logger.info(f"DashScope Embedding 服务初始化: {model_name}, batch={batch_size}, 并发={max_threads}")

    # ─────────────────────────────────────────────────────────────────────
    # 异步版本（检索路径：单个 query，不需要线程池）
    # ─────────────────────────────────────────────────────────────────────
    async def _embed_batch_async(
        self, client: httpx.AsyncClient, texts: List[str], chunk_ids: List[int]
    ) -> tuple:
        """
        异步发送单批次请求（带重试）。
        返回: (chunk_ids, vectors)
        """
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.post(
                    self.base_url, json=payload, headers=headers, timeout=30
                )
                if response.status_code == 429:
                    raise Exception("Rate Limit")
                response.raise_for_status()
                data = response.json()
                vectors = [item.get("embedding") for item in data.get("data", [])]
                return (chunk_ids, vectors)
            except Exception as e:
                logger.warning(f"向量化批次失败 ({attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    logger.error("重试耗尽，跳过该批次。")
                    return (chunk_ids, [None] * len(texts))
                # 首次失败直接重试不等待，后续等待也控制在1秒内（embedding API足够可靠）
                await asyncio.sleep(0.5 * attempt)

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        全异步入口（事件循环不阻塞，适合检索场景）。
        并发控制：用 semaphore 限制同时发起的请求数。
        """
        if not texts:
            return []

        import asyncio
        total = len(texts)
        logger.info(f"异步向量化: 共 {total} 块, batch={self.batch_size}...")

        batches = []
        for i in range(0, total, self.batch_size):
            batches.append((texts[i:i+self.batch_size], list(range(i, i + self.batch_size))))

        final_vectors = [None] * total
        semaphore = asyncio.Semaphore(5)  # 最多同时 5 个批次

        async def _run_batch(client, b_texts, b_ids):
            async with semaphore:
                return await self._embed_batch_async(client, b_texts, b_ids)

        async with httpx.AsyncClient() as client:
            tasks = [_run_batch(client, bt, bi) for bt, bi in batches]
            results = await asyncio.gather(*tasks)

        success = 0
        for b_ids, b_vectors in results:
            for idx, vec in zip(b_ids, b_vectors):
                final_vectors[idx] = vec
                if vec is not None:
                    success += 1

        logger.info(f"向量化完成: {success}/{total}")
        return final_vectors

    # ─────────────────────────────────────────────────────────────────────
    # 同步版本（入库场景：大量文档入库，需要真正的线程并发）
    # ─────────────────────────────────────────────────────────────────────
    def _embed_batch_sync(self, texts: List[str], chunk_ids: List[int]) -> tuple:
        """同步单批次请求（入库时在线程池里跑）"""
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                import requests as _requests
                response = _requests.post(self.base_url, json=payload, headers=headers, timeout=30)
                if response.status_code == 429:
                    raise Exception("Rate Limit")
                response.raise_for_status()
                data = response.json()
                vectors = [item.get("embedding") for item in data.get("data", [])]
                return (chunk_ids, vectors)
            except Exception as e:
                logger.warning(f"同步向量化批次失败 ({attempt}/{self.max_retries}): {e}")
                if attempt == self.max_retries:
                    return (chunk_ids, [None] * len(texts))
                time.sleep(1.5 ** attempt)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        多线程版本（用于文档入库，一次处理大量 chunks）。
        对检索场景请用 embed_texts_async()。
        """
        if not texts:
            return []

        total_chunks = len(texts)
        logger.info(f"同步多线程向量化: 共 {total_chunks} 块, {self.max_threads} 线程...")

        batches = []
        for i in range(0, total_chunks, self.batch_size):
            batches.append((texts[i:i+self.batch_size], list(range(i, i+len(texts[i:i+self.batch_size])))))

        final_vectors = [None] * total_chunks
        success_count = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [
                executor.submit(self._embed_batch_sync, bt, bi)
                for bt, bi in batches
            ]
            for future in as_completed(futures):
                b_ids, b_vectors = future.result()
                for idx, vec in zip(b_ids, b_vectors):
                    final_vectors[idx] = vec
                    if vec is not None:
                        success_count += 1

        logger.info(f"向量化完成: {success_count}/{total_chunks}")
        return final_vectors

    # ─────────────────────────────────────────────────────────────────────
    # 检索专用：单 query 高效查向量
    # ─────────────────────────────────────────────────────────────────────
    async def embed_query_async(self, text: str) -> List[float]:
        """异步单 query 向量化，检索专用。"""
        async with httpx.AsyncClient() as client:
            b_ids, b_vectors = await self._embed_batch_async(client, [text], [0])
            return b_vectors[0] if b_vectors and b_vectors[0] is not None else []

    # ─────────────────────────────────────────────────────────────────────
    # 兼容别名
    # ─────────────────────────────────────────────────────────────────────
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        ids, vectors = self._embed_batch_sync([text], [0])
        return vectors[0] if vectors and vectors[0] is not None else []
