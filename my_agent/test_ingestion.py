"""
测试我们的 RAG 数据清洗流水线（MVP版）
"""
import sys
from loguru import logger

# 引入我们刚才写的两个大模块
from document_loaders import route_and_load
from services.document_splitter import DocumentSplitter

def test_pipeline(file_path: str):
    logger.info(f"=== 🚀 RAG 数据入库流水线测试开始: {file_path} ===")
    
    # 【步骤 1：加载原始文档】
    try:
        raw_documents = route_and_load(file_path)
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        return

    # 【步骤 2：文档清洗与切分】
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50) # 为了测试看的明显，我把参数调小
    chunked_documents = splitter.split_documents(raw_documents)
    
    logger.info("=== 📊 切分结果预览 ===")
    
    # 打印前 3 块验证效果，看看我们的正则和断句有没有生效
    for i, chunk in enumerate(chunked_documents[:3]):  
        print(f"\n--- Chunk {i+1} (字符数: {len(chunk.page_content)}) ---")
        preview = chunk.page_content[:200].replace('\n', '\\n') + "..." if len(chunk.page_content) > 200 else chunk.page_content.replace('\n', '\\n')
        print(preview)
        print(f"元数据: {chunk.metadata}")
        
    logger.info(f"=== ✅ 测试结束：原始文档被转化为 {len(chunked_documents)} 块高质量 Chunk ===")
    
    # 下一步预告：
    # TODO: 步骤 3：调用 Embedding 模型把上面的文本转成 [0.1, 0.4, ...] 向量
    # TODO: 步骤 4：把向量和文本存入 Milvus 数据库

if __name__ == "__main__":
    import os
    # ========== 请在这里填入一个你电脑上真实的测试文件路径 ========== 
    # 找个内容稍微多点（哪怕几百字）的 .txt, .md, .docx 都可以
    test_file_path = r"E:\ragent\python_ragent\walkthrough.md"
    
    if os.path.exists(test_file_path):
        test_pipeline(test_file_path)
    else:
        logger.warning(f"请先把代码里的 test_file_path 改成真实存在的文件路径，再运行测试脚本！")
