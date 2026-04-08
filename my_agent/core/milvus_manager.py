"""
Milvus 向量库核心管理器 (Milvus Manager) - Python v2 API 版

本模块通过 pymilvus 的 MilvusClient (v2.4+) 实现，完美对齐 Java SDK 中的 v2 API 设计。
核心能力：
1. 自动维护数据模式 (Schema) 与 4096 维向量空间。
2. 支持 Dense (稠密) + Sparse (稀疏) 混合索引。
3. 提供高性能批量插入与 RRF 混合检索。
"""

try:
    from pymilvus import (
        MilvusClient,
        DataType,
        AnnSearchRequest,
        RRFRanker
    )
except ImportError:
    pass

from loguru import logger
from typing import List, Dict, Any, Callable, TypeVar

T = TypeVar("T")


class MilvusManager:
    def __init__(self, host: str = "127.0.0.1", port: str = "19530", collection_name: str = "mydoc_knowledge_base", dim: int = 1024):
        """
        初始化向量库管理器 (使用 v2 MilvusClient API)
        
        Args:
            dim: 向量维度。必须与 Qwen3-Embedding-8B 的 4096 维对齐。
        """
        self.uri = f"http://{host}:{port}"
        self.collection_name = collection_name
        self.dim = dim
        self.client = None
        
        self._connect()
        # 初始化集合逻辑不再放构造函数中自动执行，由上层通过 init_collection 显式触发更安全
        # 或者保留在此处但增加 skip_init 参数。为了保持之前逻辑，我们还是显式调用。

    @staticmethod
    def _is_stale_connection_error(exc: BaseException) -> bool:
        """gRPC 通道空闲过久或被服务端关闭后，会出现 closed channel / RPC 不可用。"""
        msg = str(exc).lower()
        return (
            "closed channel" in msg
            or "invoke rpc on closed" in msg
            or "connection reset" in msg
            or "unavailable" in msg
            or "broken pipe" in msg
        )

    def _with_reconnect(self, op_name: str, fn: Callable[[Any], T]) -> T:
        """
        使用当前 client 执行 fn(client)；若连接已失效则丢弃 client、重连后重试一次。
        """
        last_err: BaseException | None = None
        for attempt in range(2):
            if not self.client:
                self._connect()
            if not self.client:
                raise RuntimeError("Milvus 客户端未初始化，无法执行 " + op_name)
            try:
                return fn(self.client)
            except Exception as e:
                last_err = e
                if attempt == 0 and self._is_stale_connection_error(e):
                    logger.warning(f"⚠️ Milvus [{op_name}] 连接已失效，正在重连并重试一次: {e}")
                    self.client = None
                    self._connect()
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _connect(self):
        """连接 Milvus 服务 (v2 简易连接)"""
        try:
            self.client = MilvusClient(uri=self.uri)
            logger.info(f"✅ 成功连接到 Milvus 服务 (v2 Client): {self.uri}")
        except Exception as e:
            logger.error(f"❌ 连接 Milvus 失败: {e}. 请确保 docker-compose 已启动。")

    def init_collection(self, collection_name: str = None):
        """建库建表与索引核心逻辑 (v2 风格)"""
        if not self.client:
            self._connect()
        if not self.client:
            logger.error("客户端未连接，无法初始化集合。")
            return

        target_name = collection_name or self.collection_name
        dim = self.dim

        def _init(client):
            if client.has_collection(target_name):
                logger.info(f"集合 {target_name} 已存在，跳过初始化。")
                client.load_collection(target_name)
                return

            logger.info(f"====== 正在初始化全新的 Milvus v2 集合: {target_name} ======")
            schema = client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
                description="RAG 混合检索大满贯集合",
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="page_content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"},
            )
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
                nullable=False,
            )
            schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=200)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1000)

            from pymilvus import Function, FunctionType

            bm25_function = Function(
                name="text_bm25_emb",
                function_type=FunctionType.BM25,
                input_field_names=["page_content"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(bm25_function)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            index_params.add_index(field_name="category", index_type="TRIE")

            client.create_collection(
                collection_name=target_name,
                schema=schema,
                index_params=index_params,
            )
            client.load_collection(target_name)
            logger.info(f"✨ 成功创建带有混合搜索索引的 v2 集合: {target_name}")

        try:
            self._with_reconnect("init_collection", _init)
        except Exception as e:
            logger.error(f"init_collection 失败: {e}")

    def insert(self, ids: List[str], vectors: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]], collection_name: str = None):
        """
        批量插入数据 (v2 API 支持 List[Dict] 格式)
        """
        if not self.client:
            self._connect()
        if not self.client:
            raise RuntimeError("Milvus 客户端未初始化。")

        target_name = collection_name or self.collection_name
        data_to_insert = []
        for i in range(len(texts)):
            row = {
                "page_content": texts[i],
                "embedding": vectors[i],
                "category": metadatas[i].get("category", "default"),
                "source": metadatas[i].get("source", "unknown"),
            }
            data_to_insert.append(row)

        def _do(client):
            return client.insert(collection_name=target_name, data=data_to_insert)

        res = self._with_reconnect("insert", _do)
        logger.info(f"🎉 成功向 Milvus 写入 {len(data_to_insert)} 条数据快照。")
        return res

    def hybrid_search(self, query_text: str, query_dense_vec: List[float], top_k: int = 10, collection_name: str = None) -> List[Dict[str, Any]]:
        """
        混合检索核心逻辑: Dense (嵌入) + Sparse (BM25) + RRF (融合)
        """
        if not self.client:
            self._connect()
        if not self.client:
            return []

        target_name = collection_name or self.collection_name

        dense_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        dense_req = AnnSearchRequest(
            data=[query_dense_vec],
            anns_field="embedding",
            param=dense_search_params,
            limit=top_k,
        )
        sparse_search_params = {"metric_type": "BM25"}
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=top_k,
        )

        def _do(client):
            return client.hybrid_search(
                collection_name=target_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=top_k,
                output_fields=["page_content", "category", "source"],
            )

        try:
            res = self._with_reconnect("hybrid_search", _do)
        except Exception as e:
            logger.error(f"hybrid_search 失败: {e}")
            return []

        results = []
        if res and len(res) > 0:
            hits = res[0]
            for hit in hits:
                entity = hit.get("entity", {})
                results.append({
                    "id": hit.get("id"),
                    "score": hit.get("distance"),
                    "text": entity.get("page_content", ""),
                    "category": entity.get("category", ""),
                    "source": entity.get("source", ""),
                })

        logger.info(f"🔎 混合检索完成！Top-{top_k} 结果已召回。")
        return results
