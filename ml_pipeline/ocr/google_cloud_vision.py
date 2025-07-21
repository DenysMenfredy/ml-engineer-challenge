import io
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types
from ml_pipeline.ocr.base import BaseOCRProcessor, BoundingBox, OCRProcessingError, OCRResult, TextBlock

class GoogleCloudVisionOCRProcessor(BaseOCRProcessor):
    """OCR processor using Google Cloud Vision API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = vision.ImageAnnotatorClient()
        self.config = config or {}

    def get_supported_formats(self) -> List[str]:
        """Get supported image formats."""
        return ["png", "jpg", "jpeg", "tiff", "bmp", "gif", "pdf"]
    
    
    def extract_text(self, image: Union[str, bytes, Image.Image]) -> OCRResult:
        """Extract text from an image using Google Cloud Vision API."""
        try:
            if isinstance(image, str):
                # Load image from file path
                with open(image, 'rb') as f:
                    image = f.read()
            elif isinstance(image, Image.Image):
                # Convert PIL Image to bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image = buffer.getvalue()
            elif isinstance(image, bytes):
                # Already bytes, do nothing
                pass
            else:
                raise ValueError("Unsupported image type for OCR")

            # Prepare the image for Google Cloud Vision API
            vision_image = types.Image(content=image)
            feature = types.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            request = types.AnnotateImageRequest(
                image=vision_image,
                features=[feature],
                image_context={
                    "language_hints": self.config.get("language_hints", ["en"])
                }
            )

            # Send the request to the Google Cloud Vision API
            response = self.client.annotate_image(request)
            if response.error.message:
                raise OCRProcessingError(f"Google Cloud Vision API error: {response.error.message}")

            # Process the response
            text_annotations = response.full_text_annotation
            if not text_annotations:
                raise OCRProcessingError("No text annotations found in the image.")

            # Extract text and bounding boxes
            text = text_annotations.text
            blocks = []
            overall_confidence = 0.0
            count = 0

            for page_idx, page in enumerate(text_annotations.pages, start=1):
                for block in page.blocks:
                    # Skip non-text blocks
                    if block.block_type != vision.Block.BlockType.TEXT:
                        continue

                    # Get bounding box coordinates
                    vertices = block.bounding_box.vertices
                    if len(vertices) >= 4:
                        bbox = BoundingBox(
                            x = float(vertices[0].x),
                            y = float(vertices[0].y),
                            width = float(vertices[2].x - vertices[0].x),
                            height = float(vertices[2].y - vertices[0].y),
                        )
                    else:
                        bbox = None

                    # Aggregate text and confidence from all blocks
                    block_text = ""
                    block_confidence = 0.0
                    word_count = 0

                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                            if word.confidence:
                                block_confidence += word.confidence
                                word_count += 1

                    block_confidence /= word_count if word_count > 0 else 0.0

                    # Create text block
                    text_block = TextBlock(
                        text=block_text.strip(),
                        confidence=block_confidence,
                        bounding_box=bbox,
                        page_number=page_idx,  # Use the loop index as the page number
                        block_type="text",
                    )

                    blocks.append(text_block)
                    overall_confidence += block_confidence
                    count += 1

            # Calculate average confidence
            average_confidence = overall_confidence / count if count > 0 else 0.0

            return OCRResult(
                text=text,
                blocks=blocks,
                confidence=average_confidence,
                page_count=len(text_annotations.pages),
                metadata={
                    "language_hints": self.config.get("language_hints", ["en"]),
                    "detected_languages": [
                        lang.language_code for lang in text_annotations.pages[0].property.detected_languages
                    ] if text_annotations.pages else []
                },
                raw_response=response._pb
            )

        except Exception as e:
            raise OCRProcessingError(f"Error extracting text from image: {str(e)}")
