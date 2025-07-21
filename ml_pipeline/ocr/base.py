from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image

@dataclass
class BoundingBox:
    """Represents a bounding box for text regions with coordinates and dimensions."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class TextBlock:
    """Represents a text block with bounding box and text content."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    page_number: int = 1
    block_type: str = "text"  # text, table, image, etc.

    def __post_init__(self):
        """Validate confidence score."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

@dataclass
class OCRResult:
    """Represents the result of OCR processing."""
    text: str
    confidence: float
    page_count: int = 1
    blocks: Optional[List[TextBlock]] = None
    metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None

class OCRProcessingError(Exception):
    """Exception raised for OCR processing errors."""
    pass

class BaseOCRProcessor(ABC):
    """Abstract base class for OCR processors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return the list of supported file formats."""
        pass

    @abstractmethod
    def extract_text(self, image: Union[str, bytes, Image.Image]) -> OCRResult:
        """Extract text from an image using OCR."""
        pass