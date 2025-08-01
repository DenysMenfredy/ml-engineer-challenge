import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
import os
import chromadb
from django.conf import settings
from rest_framework.parsers import MultiPartParser, FormParser
from ml_pipeline.ocr.google_cloud_vision import GoogleCloudVisionOCRProcessor
from ml_pipeline.ocr.pipeline import OCRPipeline
from ml_pipeline.dataset.utils import clean_text
from services.logger import logger
from ml_pipeline.entity_extractor.extractor import EntityExtractor
import json

# Preload the model ONCE at module load, using Ollama for entity extraction with gemma3:1b
enity_extractor = EntityExtractor(use_ollama=True, ollama_model="gemma3:1b")

class DocumentProcessingView(APIView):
    """API view to process a single document, identify its type, and extract entities."""
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        logger.info(f"Request: {request}")
        file = request.FILES.get('file')
        logger.info(f"File: {file.name}")
        if not file:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the file temporarily
        safe_filename = f"{uuid.uuid4().hex}.{file.name.split('.')[-1]}"
        relative_path = os.path.join('files', safe_filename)  # 'files/uuid.jpg'
        file_path = default_storage.save(relative_path, file)
        absolute_file_path = os.path.join(settings.FILES_ROOT, safe_filename)
        
        try:
            ocr_pipeline = OCRPipeline(processor=GoogleCloudVisionOCRProcessor(config={"language_hints": ["en"]}))
            ocr_result = ocr_pipeline.process_file(absolute_file_path)
            cleaned_text = clean_text(ocr_result.text)

            # Query the ChromaDB for similar documents
            client = chromadb.PersistentClient(path=os.path.join(settings.BASE_DIR, "db"), 
                        settings=chromadb.Settings(allow_reset=True, 
                        persist_directory=os.path.join(settings.BASE_DIR, "db"), is_persistent=True))
            collection = client.get_or_create_collection(name="documents")
            results = collection.query(
                query_texts=[cleaned_text],
                n_results=5,
            )
            predicted_type = max(
                set(results["metadatas"][0][i]["class"] for i in range(len(results["metadatas"][0]))),
                key=lambda x: results["metadatas"][0].count(x)
            )
            logger.info(f"[DocumentProcessingView] Predicted type: {predicted_type}")

            # Remove per-request instantiation
            # entity_extractor = EntityExtractor()
            entities = enity_extractor.extract_entities(cleaned_text, predicted_type)
            logger.info(f"[DocumentProcessingView] Entities: {entities}")

            # Store processed document, type, and entities in ChromaDB
            collection.upsert(
                documents=[cleaned_text],
                metadatas=[{
                    "class": predicted_type,
                    "filename": file.name,
                    "confidence": ocr_result.confidence,
                    "entities": json.dumps(entities)
                }],
                ids=[file.name]
            )

            # Response
            response = {
                "filename": file.name,
                "document_type": predicted_type,
                "text": cleaned_text,
                "entities": entities,
                "confidence": ocr_result.confidence
            }

            return Response(response, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            if file_path:
                default_storage.delete(file_path)               

