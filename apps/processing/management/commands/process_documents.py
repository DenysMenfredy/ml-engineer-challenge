"""Django command to process documents"""

from django.core.management.base import BaseCommand
import os

from ml_pipeline.dataset.generator import TextDatasetGenerator
from ml_pipeline.ocr.google_cloud_vision import GoogleCloudVisionOCRProcessor
from ml_pipeline.ocr.pipeline import OCRPipeline
from services.logger import logger

class Command(BaseCommand):
    """Django management command to process documents and upsert into ChromaDB."""
    help = "Process documents in class-based subfolders and upsert into ChromaDB"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocr_processor = GoogleCloudVisionOCRProcessor()
        self.ocr_pipeline = OCRPipeline(processor=self.ocr_processor)

    def add_arguments(self, parser):
        parser.add_argument(
            "--input_dir",
            type=str,
            required=True,
            help="Directory containing document images in class-based subfolders"
        )

    def handle(self, *args, **options):
        input_dir = options["input_dir"]
        logger.info(f"input_dir: {input_dir}")

        if not os.path.isdir(input_dir):
            self.stderr.write(self.style.ERROR(f"Input directory {input_dir} does not exist"))
            return

        try:
            generator = TextDatasetGenerator(
                input_dir=input_dir,
                config={"language_hints": ["en"], "ocr_processor_pipeline": self.ocr_pipeline}
            )
            generator.generate()
            self.stdout.write(self.style.SUCCESS(f"Successfully processed documents from {input_dir}"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error processing documents: {str(e)}"))