"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from typing import List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service


# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器，使用 MilvusClient（非 ORM）直接操作"""

    def __init__(self):
        """初始化向量存储管理器"""
        self._client = None
        self.collection_name = COLLECTION_NAME

    def init_vector_store(self):
        """显式初始化 Milvus 客户端，通常在应用启动时调用"""
        if self._client is not None:
            return
        self._initialize_client()

    def _initialize_client(self):
        """内部初始化 Milvus 客户端"""
        try:
            from pymilvus import MilvusClient

            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            # 确保 collection 已加载到内存
            if self._client.has_collection(self.collection_name):
                logger.info(
                    f"Milvus 客户端初始化成功: {config.milvus_host}:{config.milvus_port}, "
                    f"collection: {self.collection_name}"
                )
            else:
                logger.warning(f"Collection '{self.collection_name}' 不存在，请先创建")

        except Exception as e:
            logger.error(f"Milvus 客户端初始化失败: {e}")
            raise

    @property
    def client(self):
        """获取 MilvusClient 实例"""
        if self._client is None:
            self.init_vector_store()
        return self._client

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid

            start_time = time.time()

            # 1. 提取文本并生成 embedding
            texts = [doc.page_content for doc in documents]
            embeddings = vector_embedding_service.embed_documents(texts)

            # 2. 生成 IDs
            ids = [str(uuid.uuid4()) for _ in documents]

            # 3. 构建 entities
            entities = []
            for i, doc in enumerate(documents):
                entities.append({
                    "id": ids[i],
                    "vector": embeddings[i],
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })

            # 4. 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=entities,
            )

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            # 使用 JSON 路径查询语法
            expr = f'metadata["_source"] == "{file_path}"'
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=expr,
            )
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_client(self):
        """获取 MilvusClient 实例"""
        return self.client

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            # 1. 查询文本向量化
            query_embedding = vector_embedding_service.embed_query(query)

            # 2. 搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["content", "metadata"],
            )

            # 3. 转换为 Document
            docs = []
            if results and len(results) > 0:
                for hit in results[0]:
                    doc = Document(
                        page_content=hit.get("entity", {}).get("content", ""),
                        metadata=hit.get("entity", {}).get("metadata", {}),
                    )
                    docs.append(doc)

            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs

        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局单例
vector_store_manager = VectorStoreManager()
