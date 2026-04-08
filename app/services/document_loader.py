"""统一文档加载器 - 多格式文档文本提取

支持格式：
- 纯文本：.txt → 直接读取
- Markdown：.md → 直接读取
- Word 文档：.docx → python-docx 解析
- PDF：.pdf → PyMuPDF 矢量提取，备选阿里云 OCR
- 图片：.png / .jpg / .jpeg / .bmp / .gif → 阿里云 OCR 识别

使用方式：
    loader = DocumentLoader()
    content = loader.load("文档路径")  # str：纯文本
    contents = loader.load_multi("文档路径")  # list[str]：多页文档各页文本
"""

from pathlib import Path
from typing import Literal, Optional

from loguru import logger

LoadResult = str  # 单格式返回文本
MultiPageResult = list[str]  # 多页格式（PDF）返回每页文本


class DocumentLoader:
    """统一文档加载器"""

    # 文本类格式（直接读取）
    TEXT_EXTS = {".txt", ".md", ".markdown"}
    # Word 格式
    DOCX_EXTS = {".docx"}
    # PDF 格式（矢量优先，OCR 兜底）
    PDF_EXTS = {".pdf"}
    # 图片格式（必须 OCR）
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    def __init__(self):
        self._ocr_service = None  # 延迟初始化

    @property
    def ocr_service(self):
        """延迟加载 OCR 服务（仅图片/PDF 需要）"""
        if self._ocr_service is None:
            from app.services.ocr_service import get_ocr_service
            self._ocr_service = get_ocr_service()
        return self._ocr_service

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def load(self, file_path: str) -> LoadResult:
        """
        加载单个文档，返回纯文本。

        支持所有格式；PDF / 图片统一返回拼接后的文本。
        如果需要按页分开处理，请使用 load_multi()。

        Args:
            file_path: 文件绝对路径或相对路径

        Returns:
            str: 提取的纯文本内容

        Raises:
            ValueError: 不支持的文件格式或文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()

        if ext in self.TEXT_EXTS:
            return self._load_text(path)
        elif ext in self.DOCX_EXTS:
            return self._load_docx(path)
        elif ext in self.PDF_EXTS:
            pages = self.load_multi(file_path)
            return "\n\n".join(pages)
        elif ext in self.IMAGE_EXTS:
            return self._load_image(path)
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持的格式: {self.TEXT_EXTS | self.DOCX_EXTS | self.PDF_EXTS | self.IMAGE_EXTS}"
            )

    def load_multi(self, file_path: str) -> MultiPageResult:
        """
        加载多页文档，按页返回文本列表。

        适用场景：需要对 PDF 每页单独分块时使用。

        Args:
            file_path: 文件路径

        Returns:
            list[str]: 每页文本列表。PDF 返回每页；其余格式返回单元素列表。
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in self.TEXT_EXTS:
            return [self._load_text(path)]
        elif ext in self.DOCX_EXTS:
            # Word 文档暂不支持分页，返回单元素
            return [self._load_docx(path)]
        elif ext in self.PDF_EXTS:
            return self._load_pdf(path)
        elif ext in self.IMAGE_EXTS:
            return [self._load_image(path)]
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def is_supported(self, file_path: str) -> bool:
        """判断文件格式是否支持加载"""
        ext = Path(file_path).suffix.lower()
        return ext in (self.TEXT_EXTS | self.DOCX_EXTS | self.PDF_EXTS | self.IMAGE_EXTS)

    def get_ext_category(self, file_path: str) -> Literal["text", "docx", "pdf", "image", "unsupported"]:
        """返回文件所属类别"""
        ext = Path(file_path).suffix.lower()
        if ext in self.TEXT_EXTS:
            return "text"
        if ext in self.DOCX_EXTS:
            return "docx"
        if ext in self.PDF_EXTS:
            return "pdf"
        if ext in self.IMAGE_EXTS:
            return "image"
        return "unsupported"

    # -------------------------------------------------------------------------
    # 内部加载方法
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_text(path: Path) -> str:
        """加载纯文本文件"""
        try:
            content = path.read_text(encoding="utf-8")
            logger.debug(f"[Loader] 文本文件加载: {path.name}, 字符数={len(content)}")
            return content
        except UnicodeDecodeError:
            # 兜底 GBK 编码（中文 Windows 常见）
            content = path.read_text(encoding="gbk")
            logger.warning(f"[Loader] 文本文件 {path.name} 使用 GBK 编码读取")
            return content

    @staticmethod
    def _load_docx(path: Path) -> str:
        """加载 Word 文档（.docx）"""
        try:
            from docx import Document
            doc = Document(str(path))
            paragraphs: list[str] = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            content = "\n".join(paragraphs)
            logger.debug(f"[Loader] Word 文档加载: {path.name}, 字符数={len(content)}")
            return content
        except Exception as e:
            logger.error(f"[Loader] Word 文档加载失败: {path.name}, 错误: {e}")
            raise RuntimeError(f"Word 文档加载失败: {e}") from e

    def _load_pdf(self, path: Path) -> list[str]:
        """加载 PDF（矢量文字优先，OCR 兜底）"""
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(str(path))
            page_texts: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                if not text:
                    # 该页无矢量文字，降级到 OCR
                    logger.debug(f"[Loader] PDF 第 {page_num + 1} 页无矢量文字，降级 OCR: {path.name}")
                    try:
                        page_bytes = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
                        text = self.ocr_service.recognize_image_bytes(page_bytes, mime_type="image/png")
                    except Exception as ocr_err:
                        logger.warning(f"[Loader] PDF 第 {page_num + 1} 页 OCR 也失败: {ocr_err}")
                        text = ""  # OCR 失败则该页留空，不阻塞其他页

                page_texts.append(text)
                logger.debug(f"[Loader] PDF 第 {page_num + 1}/{len(doc)} 页加载完成, 字符数={len(text)}")

            doc.close()
            logger.info(f"[Loader] PDF 加载完成: {path.name}, 总页数={len(page_texts)}")
            return page_texts

        except Exception as e:
            logger.error(f"[Loader] PDF 加载失败: {path.name}, 错误: {e}")
            raise RuntimeError(f"PDF 加载失败: {e}") from e

    def _load_image(self, path: Path) -> str:
        """加载图片（OCR 识别）"""
        try:
            text = self.ocr_service.recognize_image(str(path))
            logger.info(f"[Loader] 图片 OCR 识别完成: {path.name}, 字符数={len(text)}")
            return text
        except Exception as e:
            logger.error(f"[Loader] 图片 OCR 识别失败: {path.name}, 错误: {e}")
            raise RuntimeError(f"图片 OCR 识别失败: {e}") from e


# 全局单例
document_loader = DocumentLoader()
