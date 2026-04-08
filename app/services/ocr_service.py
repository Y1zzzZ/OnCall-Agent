"""阿里云 OCR 服务 - 识别图片/PDF 中的文字

使用阿里云 OCR API（通用文字识别），
支持本地图片和 PDF 逐页识别，返回纯文本供后续分块检索。

接入方式：
1. 安装依赖：pip install alibabacloud_ocr_api2021
2. 在 .env 中配置 ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET / ALIYUN_REGION_ID
3. 调用 OcrService.recognize_image() 或 OcrService.recognize_pdf()
"""

import base64
import io
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import config


class OcrService:
    """阿里云 OCR 服务"""

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        region_id: Optional[str] = None,
    ):
        """
        Args:
            access_key_id:     阿里云 AccessKey ID（None 时从 config 读取）
            access_key_secret: 阿里云 AccessKey Secret
            region_id:         阿里云 Region，如 cn-shanghai
        """
        self.access_key_id = access_key_id or config.aliyun_access_key_id
        self.access_key_secret = access_key_secret or config.aliyun_access_key_secret
        self.region_id = region_id or config.aliyun_region_id

        if not self.access_key_id or not self.access_key_secret:
            raise ValueError(
                "阿里云 OCR 未配置：ALIYUN_ACCESS_KEY_ID / ALIYUN_ACCESS_KEY_SECRET 不能为空"
            )

        self._client = self._init_client()
        logger.info(
            f"阿里云 OCR 服务初始化完成，Region={self.region_id}"
        )

    def _init_client(self):
        """初始化阿里云 OCR 客户端"""
        from alibabacloud_ocr_api2021 import Client
        from alibabacloud_tea_openapi import models as open_api_models
        from alibabacloud_tea_util import models as util_models

        config_obj = open_api_models.Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            region_id=self.region_id,
        )
        return Client(
            config=config_obj,
            options=util_models.RuntimeOptions(),
        )

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------

    def recognize_image(self, image_path: str) -> str:
        """
        识别单张本地图片中的文字

        Args:
            image_path: 图片文件路径（支持 .png / .jpg / .jpeg / .bmp / .gif）

        Returns:
            str: 识别出的纯文本内容

        Raises:
            ValueError: 图片不存在或格式不支持
            RuntimeError: OCR 调用失败
        """
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"图片文件不存在: {image_path}")

        ext = path.suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            raise ValueError(f"不支持的图片格式: {ext}，仅支持 png/jpg/jpeg/bmp/gif/webp")

        try:
            image_bytes = path.read_bytes()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            from alibabacloud_ocr_api2021 import models

            request = models.RecognizeCharacterRequest(
                image_url=f"data:image/{ext.lstrip('.')};base64,{image_base64}",
            )

            response = self._client.recognize_character(request)

            text = self._extract_text(response)
            logger.info(f"[OCR] 图片识别完成: {path.name}, 字符数={len(text)}")
            return text

        except Exception as e:
            logger.error(f"[OCR] 图片识别失败: {image_path}, 错误: {e}")
            raise RuntimeError(f"OCR 识别失败: {e}") from e

    def recognize_pdf(self, pdf_path: str) -> list[str]:
        """
        识别 PDF 文件每一页的文字

        Args:
            pdf_path: PDF 文件路径

        Returns:
            list[str]: 每页识别结果列表，长度等于 PDF 页数。
                       空页返回空字符串。
        """
        import fitz  # PyMuPDF

        path = Path(pdf_path)
        if not path.exists():
            raise ValueError(f"PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(path))
            page_texts: list[str] = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # 先尝试直接提取文本（矢量文字 PDF）
                text = page.get_text()

                if not text.strip():
                    # 如果该页无矢量文字，渲染为图片后走 OCR
                    logger.debug(f"[OCR] PDF 第 {page_num + 1} 页无矢量文字，尝试 OCR")
                    page_bytes = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
                    image_base64 = base64.b64encode(page_bytes).decode("utf-8")

                    from alibabacloud_ocr_api2021 import models

                    request = models.RecognizeCharacterRequest(
                        image_url=f"data:image/png;base64,{image_base64}",
                    )
                    text = self._extract_text(self._client.recognize_character(request))

                page_texts.append(text.strip())
                logger.debug(f"[OCR] PDF 第 {page_num + 1}/{len(doc)} 页识别完成, 字符数={len(text)}")

            doc.close()
            logger.info(f"[OCR] PDF 识别完成: {path.name}, 总页数={len(page_texts)}")
            return page_texts

        except Exception as e:
            logger.error(f"[OCR] PDF 识别失败: {pdf_path}, 错误: {e}")
            raise RuntimeError(f"PDF OCR 识别失败: {e}") from e

    def recognize_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """
        直接识别内存中的图片字节流

        Args:
            image_bytes: 图片二进制内容
            mime_type:  MIME 类型，如 image/png / image/jpeg

        Returns:
            str: 识别出的纯文本内容
        """
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            ext = mime_type.split("/")[-1]  # png / jpeg / ...

            from alibabacloud_ocr_api2021 import models

            request = models.RecognizeCharacterRequest(
                image_url=f"data:image/{ext};base64,{image_base64}",
            )

            text = self._extract_text(self._client.recognize_character(request))
            logger.debug(f"[OCR] Bytes 图片识别完成, 字符数={len(text)}")
            return text

        except Exception as e:
            logger.error(f"[OCR] Bytes 图片识别失败: {e}")
            raise RuntimeError(f"OCR 识别失败: {e}") from e

    # -------------------------------------------------------------------------
    # 内部工具
    # -------------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """
        从阿里云 OCR 响应中提取纯文本

        Args:
            response: 阿里云 OCR API 响应对象

        Returns:
            str: 拼接后的纯文本，多行之间用换行符分隔
        """
        # 阿里云 OCR 响应 body.data.data 是列表，每项为 dict 含 text 字段
        data = getattr(response, "data", None)
        if data is None:
            return ""

        records = getattr(data, "data", None)
        if not records:
            return ""

        lines: list[str] = []
        for record in records:
            text = getattr(record, "text", "") or ""
            if text.strip():
                lines.append(text.strip())

        return "\n".join(lines)


# 全局单例（延迟初始化，首次使用时才建）
_ocr_service: OcrService | None = None


def get_ocr_service() -> OcrService:
    """获取 OcrService 全局单例"""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OcrService()
    return _ocr_service
