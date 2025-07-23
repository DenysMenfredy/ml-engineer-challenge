
import io
from typing import Optional, Union
from PIL import Image
import numpy as np
from ml_pipeline.ocr.base import BaseOCRProcessor, OCRResult
from ml_pipeline.ocr.tesseract import TesseractOCRProcessor
from pdf2image import convert_from_path
import os
from services.logger import logger
class OCRPipeline:
    """OCR Pipeline class."""

    def __init__(self, processor: BaseOCRProcessor):
        self.processor = processor

    def _is_pdf(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')

    def _pdf_to_images(self, pdf_path: str):
        return convert_from_path(pdf_path)

    def process_file(self, image: Union[str, bytes, Image.Image]) -> OCRResult:
        """Process a file and return the OCR result. Handles PDF by splitting into images."""
        if isinstance(image, str) and self._is_pdf(image):
            # PDF: convert each page to image and OCR each
            images = self._pdf_to_images(image)
            logger.info(f"[OCRPipeline] PDF to images conversion completed. Number of pages: {len(images)}")
            all_text = []
            all_blocks = []
            confidences = []
            for img in images:
                ocr_result = self.processor.extract_text(img)
                all_text.append(ocr_result.text)
                if hasattr(ocr_result, 'blocks'):
                    all_blocks.extend(getattr(ocr_result, 'blocks', []))
                confidences.append(getattr(ocr_result, 'confidence', 0.0))
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            logger.info(f"[OCRPipeline] OCR result completed. Number of pages: {len(images)}, Average confidence: {avg_conf}")
            return OCRResult(
                text="\n".join(all_text),
                confidence=avg_conf,
                page_count=len(images),
                blocks=all_blocks,
                metadata={
                    "type": "pdf"
                },
                raw_response={'pages': len(images)},
            )

        processed_data = image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        
        result = self.processor.extract_text(image)
        return result


if __name__ == "__main__":
    # Example usage for testing
    import sys
    from ml_pipeline.ocr.tesseract import TesseractOCRProcessor

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    processor = TesseractOCRProcessor()
    pipeline = OCRPipeline(processor)
    result = pipeline.process_file(file_path)
    print("--- OCR Result ---")
    print("Text:\n", result.text)
    print("Confidence:", result.confidence)
    print("Metadata:", result.metadata)

