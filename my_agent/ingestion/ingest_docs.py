"""
文档入库脚本 (Ingestion Script)

功能：
1. 遍历本地目录加载文档 (Text/Markdown)
2. 使用系统内置的 DocumentSplitter 进行智能分词
3. 使用 SiliconFlowEmbeddingService 进行批量向量化 (Qwen3-Embedding)
4. 将向量与元数据存入 Milvus 数据库 (Dense + Sparse 混合架构)
"""

import os
import uuid
from loguru import logger

# 核心导入
from my_agent.services.document_splitter import DocumentSplitter
from my_agent.services.embedding_service import DashScopeEmbeddingService
from my_agent.core.milvus_manager import MilvusManager

class IngestionPipeline:
    def __init__(self):
        self.splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        self.embed_service = DashScopeEmbeddingService()
        self.milvus = MilvusManager(dim=1024) # 必须对齐 text-embedding-v3 的 1024 维
        
    def load_local_docs(self, folder_path: str):
        """遍历文件夹加载文本数据"""
        docs = []
        if not os.path.exists(folder_path):
            logger.error(f"路径不存在: {folder_path}")
            return docs
            
        for filename in os.listdir(folder_path):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs.append({
                        "content": content,
                        "metadata": {"source": filename, "type": "local_file"}
                    })
        return docs

    def run_pipeline(self, folder_path: str, collection_name: str = None):
        """执行完整入库流程"""
        # 确保集合已初始化
        self.milvus.init_collection(collection_name)
        
        # 1. 加载
        raw_docs = self.load_local_docs(folder_path)
        if not raw_docs:
            logger.warning("没有找到可入库的文档。")
            return

        all_chunks = []
        all_metadatas = []
        
        # 2. 切分
        for doc in raw_docs:
            chunks = self.splitter.split_text(doc["content"])
            for chunk in chunks:
                all_chunks.append(chunk)
                # 每个 chunk 携带来源元数据
                all_metadatas.append(doc["metadata"])
        
        logger.info(f"切分完成: {len(raw_docs)} 个原始文档 -> {len(all_chunks)} 个切片")

        # 3. 向量化 (批量执行)
        vectors = self.embed_service.embed_documents(all_chunks)
        
        # 4. 入库 Milvus
        # 生成唯一的 Chunk ID (可以是 UUID 字符串的哈希，或者简单的自增)
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(all_chunks))]
        
        self.milvus.insert(
            ids=chunk_ids,
            vectors=vectors,
            texts=all_chunks,
            metadatas=all_metadatas,
            collection_name=collection_name
        )
        
        logger.success(f"🎉 入库成功！共 {len(chunk_ids)} 条向量已存入 Milvus。")

if __name__ == "__main__":
    # 使用示例
    # 请先在项目根目录创建一个 data/ 文件夹，并放入一些 .txt 文章
    pipeline = IngestionPipeline()
    data_dir = os.path.join(os.getcwd(), "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"已创建默认目录 {data_dir}，请将测试文档放入其中后重新运行。")
    else:
        pipeline.run_pipeline(data_dir)
