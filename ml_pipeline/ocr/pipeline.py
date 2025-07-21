
import io
from typing import Optional, Union
from PIL import Image
import numpy as np
from ml_pipeline.ocr.base import BaseOCRProcessor, OCRResult
from ml_pipeline.ocr.tesseract import TesseractOCRProcessor
from ml_pipeline.ocr.preprocessor import OCRPreprocessor

class OCRPipeline:
    """OCR Pipeline class."""

    def __init__(self, processor: BaseOCRProcessor, preprocessor: Optional[OCRPreprocessor] = None):
        self.processor = processor
        self.preprocessor = preprocessor

    def process_file(self, image: Union[str, bytes, Image.Image]) -> OCRResult:
        """Process a file and return the OCR result."""
        processed_data = image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        
        if self.preprocessor:
            # convert PIL.Image to np.ndarray
            image_array = np.array(image)
            processed_image, processing_info = self.preprocessor.preprocess_pipeline(image_array)
            # Convert back to PIL Image
            if processed_image is not None:
                image = Image.fromarray(processed_image)

        
        result = self.processor.extract_text(image)
        return result

