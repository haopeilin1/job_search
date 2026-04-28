"""
OCR 服务 —— 图片文字提取，支持多后端切换

设计原则：
1. 抽象接口：OCRBackend，统一入参（image_bytes）和出参（text, confidence）
2. 多后端支持：
   - VisionLLMBackend：调用多模态 LLM（gpt-4o 等），通用但成本高
   - LocalOCRBackend：本地 OCR（rapidocr / cnocr / paddleocr），快且免费
3. 混合策略：LocalOCR 为主，低置信度时自动 fallback 到 VisionLLM
4. 配置驱动：通过 settings.OCR_BACKEND 切换后端

使用示例：
    from app.services.ocr_service import OCRService
    service = OCRService()
    result = await service.extract(image_bytes, filename="jd.jpg")
    print(result.text)        # 提取的文字
    print(result.confidence)  # 置信度 0.0-1.0
    print(result.backend)     # 实际使用的后端
"""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.core.llm_client import LLMClient
from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class OCRResult:
    """OCR 提取结果"""
    text: str
    confidence: float       # 0.0-1.0，整体置信度
    backend: str            # 实际使用的后端名称
    raw_metadata: dict = None  # 原始元数据（如各行的置信度）


# ═══════════════════════════════════════════════════════
# 2. 抽象后端接口
# ═══════════════════════════════════════════════════════

class OCRBackend(ABC):
    """OCR 后端抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def extract(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        """从图片中提取文字"""
        pass


# ═══════════════════════════════════════════════════════
# 3. Vision LLM 后端（通用，成本高）
# ═══════════════════════════════════════════════════════

_VISION_OCR_PROMPT = """你是一位专业的文档文字提取助手。

任务：仔细查看用户上传的图片，将图片中的所有文字内容完整提取出来。

要求：
1. 尽可能完整地提取图片中的全部文字，不要遗漏任何信息
2. 保持原文的段落结构和列表格式
3. 如果图片中有表格，用文本方式描述表格内容
4. 只输出提取到的文字内容，不要添加任何解释或评论
5. 如果某些文字因模糊无法识别，用 [模糊] 标记

请直接输出提取到的文本："""


class VisionLLMBackend(OCRBackend):
    """
    基于多模态 LLM 的 OCR 后端。

    优点：通用性强，各种图片格式都能处理，无需额外依赖
    缺点：每次调用消耗 LLM token，成本高，延迟大（几百ms~几秒）
    """

    @property
    def name(self) -> str:
        return "vision_llm"

    async def extract(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        # 推断 MIME 类型
        content_type = self._guess_mime_type(filename)

        # base64 编码
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{content_type};base64,{b64}"

        # 调用 vision LLM
        vision_llm = LLMClient.from_vision_config()
        extracted_text = await vision_llm.vision_chat(
            system_prompt=_VISION_OCR_PROMPT,
            image_data_uris=[data_uri],
            temperature=0.3,
            max_tokens=4096,
        )

        # Vision LLM 没有明确的置信度，假设总是高置信度
        confidence = 0.85
        if "[模糊]" in extracted_text or len(extracted_text.strip()) < 20:
            confidence = 0.5

        logger.info(f"[VisionLLMBackend] extracted {len(extracted_text)} chars | confidence={confidence:.2f}")
        return OCRResult(
            text=extracted_text.strip(),
            confidence=confidence,
            backend=self.name,
            raw_metadata={"model": settings.VISION_MODEL},
        )

    @staticmethod
    def _guess_mime_type(filename: str) -> str:
        ext = filename.split(".")[-1].lower() if "." in filename else "jpeg"
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
            "gif": "image/gif", "bmp": "image/bmp",
        }
        return mime_map.get(ext, "image/jpeg")


# ═══════════════════════════════════════════════════════
# 4. 本地 OCR 后端（快，免费，需安装依赖）
# ═══════════════════════════════════════════════════════

class LocalOCRBackend(OCRBackend):
    """
    基于本地 OCR 引擎的后端（rapidocr / cnocr / paddleocr）。

    使用说明：
    1. 安装依赖（三选一）：
       - pip install rapidocr-onnxruntime   # 推荐，轻量，基于 ONNX
       - pip install cnocr                  # 中文专用
       - pip install paddleocr              # 效果最好但最重
    2. 设置环境变量或修改 config：OCR_BACKEND="local"

    优点：本地运行，不消耗 LLM token，延迟低（几十~几百 ms）
    缺点：需要安装额外依赖和模型文件，复杂布局效果可能不如 LLM
    """

    _engine = None  # 懒加载

    @property
    def name(self) -> str:
        return "local_ocr"

    async def extract(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        engine = self._get_engine()
        if engine is None:
            raise RuntimeError(
                "本地 OCR 引擎未初始化。请安装依赖：\n"
                "  pip install rapidocr-onnxruntime\n"
                "或：pip install cnocr\n"
                "或：pip install paddleocr"
            )

        # 调用本地 OCR
        try:
            result = engine(image_bytes)
            text = "\n".join(line[1] for line in result)
            confidences = [line[2] for line in result]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(f"[LocalOCRBackend] extracted {len(result)} lines | avg_conf={avg_conf:.2f}")
            return OCRResult(
                text=text.strip(),
                confidence=avg_conf,
                backend=self.name,
                raw_metadata={"line_count": len(result), "confidences": confidences},
            )
        except Exception as e:
            logger.error(f"[LocalOCRBackend] OCR 失败: {e}")
            raise

    def _get_engine(self):
        """懒加载 OCR 引擎，支持多种后端"""
        if self._engine is not None:
            return self._engine

        # 尝试 rapidocr（推荐，最轻量，无需 PyTorch）
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            logger.info("[LocalOCRBackend] 使用 rapidocr-onnxruntime")
            return self._engine
        except ImportError:
            pass

        # 尝试 pytesseract（轻量，依赖系统 Tesseract）
        try:
            import pytesseract
            from PIL import Image
            import io
            import os

            # 自动查找 Tesseract 安装路径（Windows 常见路径）
            tesseract_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
                os.path.expanduser(r"~\Tesseract-OCR\tesseract.exe"),
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"[LocalOCRBackend] 找到 Tesseract: {path}")
                    break

            # 验证 Tesseract 可用
            version = pytesseract.get_tesseract_version()
            logger.info(f"[LocalOCRBackend] 使用 pytesseract (Tesseract {version})")

            # 包装为统一接口
            self._engine = self._wrap_pytesseract(pytesseract, Image, io)
            return self._engine
        except Exception as e:
            logger.warning(f"[LocalOCRBackend] pytesseract 不可用: {e}")

        # 尝试 cnocr
        try:
            from cnocr import CnOcr
            self._engine = CnOcr()
            logger.info("[LocalOCRBackend] 使用 cnocr")
            return self._engine
        except ImportError:
            pass

        # 尝试 paddleocr
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            logger.info("[LocalOCRBackend] 使用 paddleocr")
            return self._engine
        except ImportError:
            pass

        logger.warning(
            "[LocalOCRBackend] 未找到任何本地 OCR 引擎。如需启用本地 OCR，请安装依赖：\n"
            "  pip install rapidocr-onnxruntime   # 推荐\n"
            "  或：pip install pytesseract        # 需额外安装 Tesseract 引擎\n"
            "  或：pip install cnocr\n"
            "  或：pip install paddleocr"
        )
        return None

    @staticmethod
    def _wrap_pytesseract(pytesseract, Image, io):
        """将 pytesseract 包装为统一接口"""
        def engine(image_bytes: bytes):
            img = Image.open(io.BytesIO(image_bytes))
            # 转换为 RGB（处理 PNG 透明通道等）
            if img.mode != "RGB":
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            # pytesseract 没有逐行置信度，返回模拟结果
            return [[(0, 0), text.strip(), 0.85]]
        return engine


# ═══════════════════════════════════════════════════════
# 5. 混合策略后端（Local 为主 + Vision LLM fallback）
# ═══════════════════════════════════════════════════════

class HybridOCRBackend(OCRBackend):
    """
    混合策略：先用本地 OCR，若置信度低则 fallback 到 Vision LLM。

    阈值配置：
    - LOCAL_CONFIDENCE_THRESHOLD = 0.65
      本地 OCR 平均置信度低于此值时，自动用 Vision LLM 重新提取
    - MIN_TEXT_LENGTH = 30
      提取文本过短（少于30字）时，也触发 fallback
    """

    LOCAL_CONFIDENCE_THRESHOLD = 0.65
    MIN_TEXT_LENGTH = 30

    def __init__(self):
        self.local = LocalOCRBackend()
        self.vision = VisionLLMBackend()

    @property
    def name(self) -> str:
        return "hybrid"

    async def extract(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        # 1. 先尝试本地 OCR
        local_result = None
        try:
            local_result = await self.local.extract(image_bytes, filename)
        except Exception as e:
            logger.warning(f"[HybridOCRBackend] 本地 OCR 失败: {e}，直接 fallback 到 Vision LLM")

        # 2. 判断是否需要 fallback
        needs_fallback = False
        if local_result is None:
            needs_fallback = True
        elif local_result.confidence < self.LOCAL_CONFIDENCE_THRESHOLD:
            needs_fallback = True
            logger.info(f"[HybridOCRBackend] 本地 OCR 置信度低({local_result.confidence:.2f})，fallback 到 Vision LLM")
        elif len(local_result.text.strip()) < self.MIN_TEXT_LENGTH:
            needs_fallback = True
            logger.info(f"[HybridOCRBackend] 本地 OCR 文本过短({len(local_result.text)} chars)，fallback 到 Vision LLM")

        if not needs_fallback:
            return local_result

        # 3. fallback 到 Vision LLM
        vision_result = await self.vision.extract(image_bytes, filename)
        vision_result.raw_metadata = {
            **(vision_result.raw_metadata or {}),
            "fallback_from": "local_ocr",
            "local_confidence": local_result.confidence if local_result else 0,
            "local_text_length": len(local_result.text) if local_result else 0,
        }
        return vision_result


# ═══════════════════════════════════════════════════════
# 6. OCR 服务入口
# ═══════════════════════════════════════════════════════

class OCRService:
    """
    OCR 服务统一入口。

    根据配置自动选择后端：
    - settings.OCR_BACKEND="vision_llm" → VisionLLMBackend
    - settings.OCR_BACKEND="local"      → LocalOCRBackend
    - settings.OCR_BACKEND="hybrid"     → HybridOCRBackend（默认推荐）
    """

    def __init__(self, backend_name: Optional[str] = None):
        self.backend_name = backend_name or getattr(settings, "OCR_BACKEND", "hybrid")
        self._backend = self._create_backend(self.backend_name)

    def _create_backend(self, name: str) -> OCRBackend:
        if name == "vision_llm":
            return VisionLLMBackend()
        elif name == "local":
            return LocalOCRBackend()
        elif name == "hybrid":
            return HybridOCRBackend()
        else:
            logger.warning(f"[OCRService] 未知后端 '{name}'，使用 hybrid")
            return HybridOCRBackend()

    async def extract(self, image_bytes: bytes, filename: str = "") -> OCRResult:
        """提取图片中的文字"""
        return await self._backend.extract(image_bytes, filename)

    @property
    def backend(self) -> str:
        return self._backend.name
