from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict, Any, Optional
from services.logger import logger
import os
import threading

class EntityExtractor:
    """
    Entity extractor using Hugging Face's dslim/bert-base-NER model (English NER).
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        logger.info(f"[EntityExtractor] Model loading started. Model: {model_name}")
        try:
            cache_dir = os.environ.get('HF_HOME', None)
            if cache_dir:
                logger.info(f"[EntityExtractor] Using HF_HOME: {cache_dir}")
            model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
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

    def extract_entities(self, text: str, document_type: str) -> Dict[str, str]:
        """
        Extract named entities from the input text.
        Args:
            text (str): The input text to extract entities from.
            document_type (str): Type of document for entity mapping
        Returns:
            Dict[str, str]: Dictionary of extracted entities
        """
        try:
            entities = self.ner_pipeline(text)
            extracted_entities = {}
            relevant_entities = self.entity_mapping.get(document_type, [])
            
            for entity in entities:
                # Map standard NER labels to your custom entity types
                entity_label = entity.get("entity_group", entity.get("entity", "")).upper()
                entity_word = entity["word"].strip()
                
                # Basic mapping from standard NER to your entities
                if entity_label in ["PER", "PERSON"]:
                    if "sender" in relevant_entities and "sender" not in extracted_entities:
                        extracted_entities["sender"] = entity_word
                    elif "recipient" in relevant_entities and "recipient" not in extracted_entities:
                        extracted_entities["recipient"] = entity_word
                    elif "author" in relevant_entities and "author" not in extracted_entities:
                        extracted_entities["author"] = entity_word
                    elif "name" in relevant_entities and "name" not in extracted_entities:
                        extracted_entities["name"] = entity_word
                        
                elif entity_label in ["ORG", "ORGANIZATION"]:
                    if "company" in relevant_entities and "company" not in extracted_entities:
                        extracted_entities["company"] = entity_word
                    elif "organization" in relevant_entities and "organization" not in extracted_entities:
                        extracted_entities["organization"] = entity_word
                    elif "institution" in relevant_entities and "institution" not in extracted_entities:
                        extracted_entities["institution"] = entity_word
                        
                elif entity_label in ["LOC", "LOCATION"]:
                    if "location" in relevant_entities and "location" not in extracted_entities:
                        extracted_entities["location"] = entity_word
                    elif "address" in relevant_entities and "address" not in extracted_entities:
                        extracted_entities["address"] = entity_word
                        
                # Add more mappings as needed
                
            return extracted_entities
            
        except Exception as e:
            logger.error(f"[EntityExtractor] Entity extraction failed: {e}")
            return {}

# Alternative approach: Global instance
_global_entity_extractor: Optional[EntityExtractor] = None
_global_lock = threading.Lock()

def get_entity_extractor(model_name: str = "dslim/bert-base-NER") -> EntityExtractor:
    """
    Get or create the global entity extractor instance.
    Thread-safe factory function.
    """
    global _global_entity_extractor
    
    if _global_entity_extractor is None:
        with _global_lock:
            if _global_entity_extractor is None:
                _global_entity_extractor = EntityExtractor(model_name)
    
    return _global_entity_extractor

# Django-specific initialization
class EntityExtractorManager:
    """
    Manager class for Django applications to handle model loading.
    """
    def __init__(self):
        self.extractor = None
        
    def get_extractor(self, model_name: str = "dslim/bert-base-NER") -> EntityExtractor:
        if self.extractor is None:
            self.extractor = EntityExtractor(model_name)
        return self.extractor

# Global manager instance
entity_manager = EntityExtractorManager()
