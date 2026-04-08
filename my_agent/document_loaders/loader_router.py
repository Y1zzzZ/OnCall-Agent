"""
文档加载路由器 (Loader Router)

使用 Python 字典映射模式，根据文件后缀名路由到特定的加载器函数。
"""
import os
from pathlib import Path
from typing import List

# 导入 LangChain 的核心 Document 数据结构
from langchain_core.documents import Document
from loguru import logger

#TODO 文件中出现图片怎么处理 pdf和word中插入的图片怎么提取文字 图片提取出来的文字 应该插在什么位置才能和内容对应起来

# ==========================================
# 第一步：准备具体的“干活函数” (The Handlers)
# ==========================================
#返回值为什么是List[Document] 因为解析文件一般不是一次性解析完 是一页一页 或者一行一行 就是一个个的Document

"""
#Document
page_content   str： 存放解析出来的原始文本内容。
metadata       dict：存放关于这段文本的所有辅助信息（来源、页码、标题等）。
"""
def load_pdf(file_path: str) -> List[Document]:
    """处理 .pdf 文件"""
    # 局部导入，因为用户不一定装了所有的包。用到谁才 import 谁，是 Python 写适配器时的好习惯
    # 懒加载
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError("缺少处理 PDF 的依赖，请先执行: pip install pypdf")

    logger.info(f"正在使用 PyPDFLoader 加载 PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    """处理 .md 文件"""
    # 对于纯文本类的 markdown，最简单稳定的方式是把它当做 Text 读取。
    # 我们后面的 Splitter 环节会专门针对 Markdown 的标题（如 #, ##）做高级的切分，
    # 所以在这个 Loader 环节，只负责把它安全地读进内存。
    from langchain_community.document_loaders import TextLoader

    logger.info(f"正在使用 TextLoader 加载 Markdown: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_word(file_path: str) -> List[Document]:
    """处理 .docx 或 .doc 文件"""
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        raise ImportError("缺少处理 Word 的依赖，请先执行: pip install docx2txt")

    logger.info(f"正在使用 Docx2txtLoader 加载 Word 文档: {file_path}")
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    """处理普通的 .txt 文件"""
    from langchain_community.document_loaders import TextLoader

    logger.info(f"正在使用 TextLoader 加载纯文本: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


# ==========================================
# 第二步：组装路由看板 (The Router Mapping)
# ==========================================

# 字典的 Key 是统一打成小写的文件后缀名，Value 是对应的加载函数。完全解耦！
LOADER_MAPPING = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".docx": load_word,
    ".doc": load_word,
    ".txt": load_text
}


# ==========================================
# 第三步：编写主入口暴露给外部 (The Entrypoint)
# ==========================================

def route_and_load(file_path: str) -> List[Document]:
    """
    智能路由加载器主入口。
    只需传入一个文件路径，它自动嗅探格式并调用相应的 LangChain Loader 解析。

    Args:
        file_path (str): 文件的绝对路径或相对路径

    Returns:
        List[Document]: 解析完毕的 LangChain 格式文档片段列表。
                        (注意：这里还是整块的文档，或是按页切分的大块，不是向量化的细粒度 Chunk)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到指定的文件: {file_path}")

    # 提取后缀并转小写 (例如 ".PDF" -> ".pdf")
    ext = Path(file_path).suffix.lower()

    # 从字典路由表中查找对应的加载函数
    loader_func = LOADER_MAPPING.get(ext)

    if loader_func is None:
        supported_exts = ", ".join(LOADER_MAPPING.keys())
        raise ValueError(f"暂不支持的文件格式: {ext}。目前仅支持: {supported_exts}")

    try:
        # 真正触发路由，执行对应的函数
        documents = loader_func(file_path)

        # 良好的实践：做一层兜底逻辑。确保每块吐出去的文档 Metadata 都带着来源路径。
        # 这样以后 RAG 检索出来，可以给用户看到是“哪篇文件”引用的。
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = file_path

        logger.info(f"成功加载文件 {file_path}，解析出 {len(documents)} 个粗粒度文档块。")
        return documents

    except Exception as e:
        logger.error(f"解析文件 {file_path} 时发生异常: {str(e)}")
        raise
