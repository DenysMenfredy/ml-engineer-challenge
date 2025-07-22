from transformers import pipeline
from typing import List, Dict, Any
from services.logger import logger
import os

class EntityExtractor:
    """
    Entity extractor using Hugging Face's dslim/bert-base-NER model (English NER).
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        logger.info(f"[EntityExtractor] Model loading started. Model: {model_name}")
        try:
            cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
            if cache_dir:
                logger.info(f"[EntityExtractor] Using TRANSFORMERS_CACHE: {cache_dir}")
            self.ner_pipeline = pipeline(
                "ner",
                tokenizer=model_name,
                model=model_name,
                aggregation_strategy="simple",
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None)
            )
            logger.info("[EntityExtractor] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[EntityExtractor] Model loading failed: {e}")
            raise
        # Entity mapping for each document class
        self.entity_mapping = {
            "advertisement": ["product_name", "price", "company", "contact_info", "date", "location"],
            "budget": ["budget_id", "department", "amount", "date", "fiscal_year", "approver"],
            "email": ["sender", "recipient", "subject", "date", "body", "signature"],
            "file_folder": ["folder_name", "creation_date", "owner", "file_count", "description"],
            "form": ["form_id", "type", "date", "status", "owner", "fields"],
            "handwritten": ["author", "date", "content", "signature"],
            "invoice": ["invoice_number", "date", "total_amount", "customer_name", 
                        "customer_address", "customer_email", "customer_phone", "organization", "client", "product"],
            "letter": ["sender", "recipient", "date", "subject", "body", "signature"],
            "memo": ["memo_id", "author", "recipient", "date", "subject", "body"],
            "news_article": ["headline", "author", "date", "location", "content", "source"],
            "presentation": ["title", "presenter", "date", "location", "slides_count", "topic"],
            "questionnaire": ["questionnaire_id", "respondent", "date", "questions", "answers"],
            "resume": ["name", "email", "phone", "address", "education", "experience", "skills"],
            "scientific_publication": ["title", "authors", "journal", "date", "volume", "issue", "pages", "doi"],
            "scientific_report": ["title", "authors", "date", "institution", "summary", "findings"],
            "specification": ["spec_id", "title", "version", "date", "author", "requirements", "description"]
        }

    def extract_entities(self, text: str, document_type: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the input text.
        Args:
            text (str): The input text to extract entities from.
        Returns:
            List[Dict[str, Any]]: List of entities with their labels and positions.
        """
        entities = self.ner_pipeline(text)
        extracted_entities = {}
        relevant_entities = self.entity_mapping.get(document_type, [])

        for entity in entities:
            if entity["entity"] in self.entity_mapping:
                extracted_entities[entity["entity"]] = entity["word"]
        return extracted_entities

