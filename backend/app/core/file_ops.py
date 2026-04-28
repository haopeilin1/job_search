"""
FileOps: 简历文件文本提取与转换工具
支持 PDF → 纯文本、DOCX → 纯文本、TXT → 纯文本、PDF → 图片(base64)
"""

import io
import base64
from typing import List


class FileOps:
    """简历文件处理工具类"""

    SUPPORTED_TEXT_EXT = {".pdf", ".docx", ".doc", ".txt"}
    SUPPORTED_IMAGE_EXT = {".pdf"}

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> str:
        """
        从文件字节中提取纯文本内容。
        支持: .pdf (pypdf), .docx/.doc (python-docx), .txt (utf-8)
        """
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if ext == "pdf":
            return cls._extract_pdf_text(file_bytes)
        elif ext in ("docx", "doc"):
            return cls._extract_docx_text(file_bytes)
        elif ext == "txt":
            return cls._extract_txt_text(file_bytes)
        else:
            # 未知格式，尝试按 txt 处理
            return cls._extract_txt_text(file_bytes)

    @staticmethod
    def _extract_pdf_text(file_bytes: bytes) -> str:
        """提取 PDF 文本，优先使用 PyMuPDF（中文支持更好），失败后用 pypdf 兜底。"""
        # 策略 1: PyMuPDF
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            texts = []
            for page in doc:
                txt = page.get_text()
                if txt:
                    texts.append(txt)
            doc.close()
            result = "\n".join(texts).strip()
            if result and not FileOps._is_cid_garbage(result):
                return result
        except Exception:
            pass  # 失败后继续尝试 pypdf

        # 策略 2: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            texts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)
            result = "\n".join(texts).strip()
            if result and not FileOps._is_cid_garbage(result):
                return result
        except Exception as e:
            raise RuntimeError(f"PDF 文本提取失败: {e}")
        return ""

    @staticmethod
    def _is_cid_garbage(text: str) -> bool:
        """检测文本是否为 PDF CID 乱码（如 /uni00000022）"""
        if not text:
            return True
        # 如果包含大量 /uni 或 /g 开头的 token，视为乱码
        import re
        cid_pattern = re.compile(r'/uni[0-9a-fA-F]{8}|/g\d+\s+\d+')
        cid_count = len(cid_pattern.findall(text))
        total_chars = len(text.replace('\n', '').replace(' ', ''))
        # CID token 占比超过 30% 视为乱码
        if total_chars == 0:
            return True
        return (cid_count * 10) / total_chars > 0.3

    @staticmethod
    def _extract_docx_text(file_bytes: bytes) -> str:
        """使用 python-docx 提取 Word 文档文本"""
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as e:
            raise RuntimeError(f"DOCX 文本提取失败: {e}")

    @staticmethod
    def _extract_txt_text(file_bytes: bytes) -> str:
        """提取纯文本文件内容，尝试多种编码"""
        for encoding in ("utf-8", "gbk", "gb2312", "utf-16", "latin-1"):
            try:
                return file_bytes.decode(encoding).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        # 兜底：用 latin-1 不会抛异常
        return file_bytes.decode("latin-1").strip()

    @classmethod
    def pdf_to_images(cls, file_bytes: bytes, dpi: int = 200) -> List[bytes]:
        """
        将 PDF 每一页转换为 PNG 图片字节列表。
        使用 PyMuPDF (fitz)，纯 Python，无需外部依赖。
        返回的列表可直接作为 base64 编码传给多模态 LLM。
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            images: List[bytes] = []
            zoom = dpi / 72  # 默认 72dpi
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
            doc.close()
            return images
        except Exception as e:
            raise RuntimeError(f"PDF 转图片失败: {e}")

    @classmethod
    def images_to_base64_data_uris(cls, images: List[bytes], mime: str = "image/png") -> List[str]:
        """将图片字节列表转为 base64 data URI 列表"""
        return [f"data:{mime};base64,{base64.b64encode(img).decode('utf-8')}" for img in images]
