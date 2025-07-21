import os
import sys
from pathlib import Path



from ml_pipeline.dataset.generator import TextDatasetGenerator
from ml_pipeline.ocr.google_cloud_vision import GoogleCloudVisionOCRProcessor
from ml_pipeline.ocr.pipeline import OCRPipeline
from ml_pipeline.ocr.preprocessor import OCRPreprocessor
from ml_pipeline.ocr.tesseract import TesseractOCRProcessor
from services.logger import logger


def main():
    logger.info("Hello from health-care-document-processing-system!")
    logger.info("Starting OCR Pipeline")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check if data directory exists
    data_dir = Path("data/docs-sm")
    if not data_dir.exists():
        logger.error(f"Error: Data directory {data_dir} does not exist!")
        return
    
    tesseract_config = {
        'language': 'eng',
        'tesseract_config': '--oem 3 --psm 6'
    }
    
    # Use Google Cloud Vision for better accuracy
    processor = GoogleCloudVisionOCRProcessor()
    preprocessor = OCRPreprocessor()
    pipeline = OCRPipeline(processor, preprocessor)
    
    # Initialize dataset generator
    dataset_generator = TextDatasetGenerator(
        input_dir=str(data_dir), 
        config={"ocr_processor_pipeline": pipeline}
    )
    
    # Generate dataset
    logger.info("Starting dataset generation...")
    dataset_generator.generate()
    logger.info("Dataset generation completed!")


if __name__ == "__main__":
    main()
