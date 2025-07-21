"""Contain classes for generating text datasets."""

import csv
import json
import os
import re
from typing import Dict, Any, Optional, Set
from abc import ABC, abstractmethod
import chromadb
from chromadb.utils import embedding_functions

from ml_pipeline.ocr.base import OCRProcessingError
from ml_pipeline.ocr.google_cloud_vision import GoogleCloudVisionOCRProcessor
from ml_pipeline.ocr.pipeline import OCRPipeline
from services.logger import logger

class BaseTextDataset(ABC):
    """Abstract base class for processing text datasets."""
    def __init__(self, input_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset processor.
        
        Args:
            input_dir: Directory containing input documents.
            config: Configuration for dataset processing.
        """
        self.input_dir = input_dir
        self.config = config or {}

    @abstractmethod
    def generate(self) -> None:
        """Process the dataset and upsert into a vector database."""
        pass

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean text for ML model input."""
        pass



class TextDatasetGenerator(BaseTextDataset):
    """Processes document images in class-based subfolders and upserts into ChromaDB."""
    def __init__(
        self,
        input_dir: str,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(input_dir, config)
        self.ocr_processor_pipeline = (config or {}).get(
            "ocr_processor_pipeline", 
            OCRPipeline(GoogleCloudVisionOCRProcessor()))
        self.chroma_client = chromadb.PersistentClient(path="db", 
                        settings=chromadb.Settings(allow_reset=True, 
                        persist_directory="db", is_persistent=True))
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )

    def clean_text(self, text: str) -> str:
        """Clean text for ML model input."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text
    
    def get_existing_document_ids(self) -> Set[str]:
        """Get set of document IDs that are already stored in ChromaDB."""
        try:
            # Get all existing documents from the collection
            results = self.collection.get()
            if results and 'ids' in results:
                return set(results['ids'])
            return set()
        except Exception as e:
            logger.error(f"Could not retrieve existing documents from ChromaDB: {e}")
            return set()
    
    def generate(self) -> None:
        """Process document images and upsert into ChromaDB, skipping already processed files."""
        supported_formats = self.ocr_processor_pipeline.processor.get_supported_formats()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Get existing document IDs to avoid reprocessing
        existing_ids = self.get_existing_document_ids()
        logger.info(f"Found {len(existing_ids)} existing documents in ChromaDB")

        for idx, class_name in enumerate(os.listdir(self.input_dir), start=1):
            logger.info(f"Processing class: {class_name}, there are {len(os.listdir(self.input_dir)) - idx} classes remaining")
            class_path = os.path.join(self.input_dir, class_name)
            if not os.path.isdir(class_path):
                logger.info(f"Skipping {class_name}: Not a directory")
                continue

            for filename in os.listdir(class_path):
                if not any(filename.lower().endswith(f".{fmt}") for fmt in supported_formats):
                    logger.info(f"Skipping {filename} in class {class_name}: Unsupported format")
                    continue

                file_path = os.path.join(class_path, filename)
                if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                    logger.info(f"Skipping {filename} in class {class_name}: File does not exist or is not readable")
                    continue

                # Check if document already exists in ChromaDB
                if filename in existing_ids:
                    logger.info(f"Skipping {filename} in class {class_name}: Already processed")
                    skipped_count += 1
                    continue

                try:
                    ocr_result = self.ocr_processor_pipeline.process_file(file_path)
                    cleaned_text = self.clean_text(ocr_result.text)

                    self.collection.upsert(
                        documents=[cleaned_text],
                        metadatas=[{
                            "class": class_name,
                            "filename": filename,
                            "confidence": ocr_result.confidence,
                            "page_count": ocr_result.page_count,
                            "detected_languages": ",".join(ocr_result.metadata.get("detected_languages", []))
                        }],
                        ids=[filename]
                    )
                    processed_count += 1
                    logger.info(f"Successfully processed: {filename} in class {class_name}")
                except OCRProcessingError as e:
                    logger.error(f"OCR Error processing {filename} in class {class_name}: {e}")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing {filename} in class {class_name}: {e}")
                    error_count += 1

        # Summary report
        logger.info(f"=== Processing Summary ===")
        logger.info(f"Newly processed documents: {processed_count}")
        logger.info(f"Skipped (already processed): {skipped_count}")
        logger.info(f"Errors encountered: {error_count}")
        
        if processed_count == 0 and skipped_count == 0:
            logger.warning("No valid files were found to process.")
        elif processed_count == 0 and skipped_count > 0:
            logger.info("All documents have already been processed.")
        else:
            logger.info(f"Successfully processed and upserted {processed_count} new documents into ChromaDB")

    



