import io
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from ml_pipeline.ocr.base import BaseOCRProcessor, BoundingBox, OCRResult, TextBlock

class TesseractOCRProcessor(BaseOCRProcessor):
    """Tesseract OCR processor."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
        except ImportError as e:
            raise ImportError(f"TesseractOCRProcessor requires pytesseract and PIL to be installed. {e}")

        self.tesseract_config = self.config.get("tesseract_config", '--oem 3 --psm 6')
        self.language = self.config.get("language", "eng")

    
    def extract_text(self, image: Union[str, bytes, Image.Image]) -> OCRResult:
        """Extract text from an image using Tesseract OCR."""
        if isinstance(image, str):
            image = self.Image.open(image)
        elif isinstance(image, bytes):
            image = self.Image.open(io.BytesIO(image))
        else:
            image = image

        data = self.pytesseract.image_to_data(
            image, 
            config=self.tesseract_config, 
            lang=self.language,
            output_type=self.pytesseract.Output.DICT
        )

        blocks = []
        full_text_parts = []

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                text = data['text'][i].strip()
                if text:
                    bbox = BoundingBox(
                        x = data['left'][i],
                        y = data['top'][i],
                        width = data['width'][i],
                        height = data['height'][i]
                    )

                    block = TextBlock(
                        text = text,
                        confidence=float(data['conf'][i]) / 100.0,
                        bounding_box=bbox,
                        block_type=self._get_block_type(data['level'][i])
                    )
                    blocks.append(block)
                    full_text_parts.append(text)

        confidences = [b.confidence for b in blocks if b.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=" ".join(full_text_parts),
            confidence=avg_confidence,
            metadata={'provider': 'tesseract', 'language': self.language},
            raw_response={'tesseract_data': data}
        )

    def _get_block_type(self, level: int) -> str:
        """Determine the block type based on Tesseract level."""
        level_map = {
            1: 'page',
            2: 'block', 
            3: 'paragraph',
            4: 'line',
            5: 'word'
        }
        return level_map.get(level, 'text')

    
    def get_supported_formats(self) -> List[str]:
        return ["jpg", "jpeg", "png", "tiff", "pdf"]

    