"""
文档切分与分析服务

此模块负责将粗粒度的 Document（比如整整一页 PDF 或一整篇 txt）
切割成细粒度的 Chunk（文本块），以便后续进行向量化和被大模型精准检索。
"""

import re
from typing import List
from langchain_core.documents import Document
from loguru import logger


#TODO 既然你已经有了清洗逻辑，一定要把“短文本合并”也加上。因为 RAG 最怕的就是“搜索结果全是碎片”。 先洗、再切、后补 补的逻辑还没写
"""
chunk_size=1000：目前主流的 Embedding 模型（比如阿里的 text-embedding 或 OpenAI 的 text-embedding-3）
最佳处理长度通常在 512 到 8192 Tokens 之间。对于中文，1000 个字符大约能包含 1-2 个完整段落。这个长度既能保证语义包含足够的上下文，
又不会因为太长导致向量的“特征被稀释”。

chunk_overlap=100：为了防止前半句和后半句彻底失去联系，100字的重叠能确保无论是哪个 chunk 被检索到，都包含这句完整的话
"""
class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):#1000
        """
        初始化文档切分器。
        
        Args:
            chunk_size (int): 每个文本块的最大长度（字符数）。模型上下文越长，这个值可以适当调大。
            chunk_overlap (int): 相邻文本块之间重叠的字符数。
                                 【切分核心理念】设置重叠是为了防止“一句话刚好被从中间劈开”导致上下文语义断裂。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 懒加载：用到才引入切分器
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # RecursiveCharacterTextSplitter 是最通用的切分器。递归切分 按照不同的符号 不断递归切分
        # 它的聪明之处在于，它会“温柔地”切分：优先按段落(\n\n)切，如果还嫌大，就按单行(\n)切，再不行才按空格切。
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # 2. 开启正则模式：这是实现“聪明切分”的前提
            is_separator_regex=True,

            # 3. 优先级策略：从大到小尝试切分
            # 逻辑：先试双换行(段落) -> 再试单换行 -> 再试中英文句号(语义) -> 最后实在不行才按空格或字符切
            # separators=[
            #     "\n\n",  # 优先保全自然段
            #     "\n",  # 退而求其次保全单行
            #     "(?<=[。！？?！][”'\"]?)",  # 中文带引号的完美断句
            #     "(?<=\. )",  # 英文安全断句（防误杀小数）
            #     " ",  # 按单词切
            #     ""  # 兜底字符切分
            # ],
            separators=[
                "\n\n",
                "\n",
                "(?<=[。！？?！][”'\"])",  # 1. 先匹配：标点+引号 (长度固定为2)
                "(?<=[。！？?！])",  # 2. 再匹配：单纯的标点 (长度固定为1)
                "(?<=\. )",  # 3. 英文断句 (长度固定为2: 点+空格)
                " ",
                ""
            ],
            # 4. 长度计算：按字符长度计算（中文 1 个字算 1 个长度）
            length_function=len,
            add_start_index=True,
        )
        logger.info(f"切分器初始化完毕: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def _clean_text(self, text: str) -> str:
        """
        [轻量级降噪]：修复 PDF 解析时常见的“回车截断”问题。
        如果一行以换行符结尾，但结尾不是句号、问号、叹号等结束语义的标点，
        我们就认为它是被排版意外截断的，强制将该换行符替换为空格。
        """
        # TODO: 进阶优化 - 引入版面分析 (Layout Analysis) 处理复杂的表格、页眉页脚、双栏排版等
        
        if not text:
            return ""

        # 1. 缩减多余的空行：把 3 个及以上的换行符缩减为 2 个（标准自然段间隔）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 2. 尝试缝合断行：如果换行符 '\n' 前面不是结束语气的标点（如句号、感叹号、问号、分号、冒号），
        # 且后面不是空行 '\n'，我们就将其合并为一个空格。（中英文标点都兼顾了）
        text = re.sub(r'([^\.。!！\?？;；:：])\n([^\n])', r'\1 \2', text)
        
        return text

    def split_text(self, text: str) -> List[str]:
        """
        [最简接口]：输入一段长文本，直接吐出切好的字符串列表。
        这对简单的 txt/md 入库非常友好。
        """
        if not text:
            return []
            
        # 1. 预处理清洗
        cleaned_text = self._clean_text(text)
        
        # 2. 调用底层切分器
        return self.base_splitter.split_text(cleaned_text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        输入大块的 documents 列表，吐出切好的小块 documents 列表。
        
        Args:
            documents: 粗粒度文档列表（例如 load_pdf 的输出）
            
        Returns:
            细小文本块列表，这些块可以直接送去转向量了！
        """
        if not documents:
            logger.warning("接收到空的文档列表，跳过切分。")
            return []
            
        logger.info(f"准备清洗并切分 {len(documents)} 块初始文档...")
        
        # [新增] 1. 预处理：清洗掉不必要的断行
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)
            
        # 2. 调用 langchain 的递归字符切分逻辑
        split_docs = self.base_splitter.split_documents(documents)
        
        logger.info(f"切分完成！原始 {len(documents)} 块大文档，被切成了 {len(split_docs)} 块 Chunk 小碎片。")
        return split_docs

