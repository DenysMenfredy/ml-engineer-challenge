from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from typing import List, Dict, Any, Optional
from services.logger import logger
import os
import threading
import re
import json
import requests

class EntityExtractor:
    """
    Entity extractor using Hugging Face's dslim/bert-base-NER model (English NER) or a prompt-based LLM.
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER", use_llm: bool = False, llm_model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", use_ollama: bool = True, ollama_model: str = "gemma3:1b"):
        self.use_llm = use_llm
        self.llm_model_name = llm_model_name
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        logger.info(f"[EntityExtractor] Model loading started. Model: {model_name}, use_llm={use_llm}, use_ollama={use_ollama}")
        try:
            cache_dir = os.environ.get('HF_HOME', None)
            if cache_dir:
                logger.info(f"[EntityExtractor] Using HF_HOME: {cache_dir}")
            if use_llm and not use_ollama:
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=llm_model_name,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0
                )
                logger.info(f"[EntityExtractor] LLM pipeline loaded: {llm_model_name}")
            elif not use_llm and not use_ollama:
                model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                logger.info("[EntityExtractor] NER pipeline loaded successfully.")
            # Ollama does not require model loading here
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
        Extract named entities from the input text using either NER/regex or LLM prompt-based extraction.
        """
        if self.use_ollama:
            prompt = f"""Extract the following fields for a {document_type} from the document text below. Return the result as a JSON object with keys for each field.\n\nDocument text:\n""" + text + """\n"""
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                result = response.json()["response"]
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    return json.loads(result[json_start:json_end])
                else:
                    logger.warning("[EntityExtractor] Ollama did not return JSON.")
                    return {}
            except Exception as e:
                logger.error(f"[EntityExtractor] Ollama extraction failed: {e}")
                return {}
        elif self.use_llm:
            # Use prompt-based LLM extraction
            prompt = f"""Extract the following fields for a {document_type} from the document text below. Return the result as a JSON object with keys for each field.

Document text:
"""
            prompt += text
            prompt += """
"""
            try:
                result = self.llm_pipeline(prompt)[0]['generated_text']
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    return json.loads(result[json_start:json_end])
                else:
                    logger.warning("[EntityExtractor] LLM did not return JSON.")
                    return {}
            except Exception as e:
                logger.error(f"[EntityExtractor] LLM extraction failed: {e}")
                return {}
        else:
            # Use NER/regex extraction (previous logic)
            try:
                entities = self.ner_pipeline(text)
                extracted_entities = {}
                relevant_entities = self.entity_mapping.get(document_type, [])

                # 1. Map standard NER entities to custom fields
                for entity in entities:
                    entity_label = entity.get("entity_group", entity.get("entity", "")).upper()
                    entity_word = entity["word"].strip()
                    if entity_label in ["PER", "PERSON"]:
                        for field in ["sender", "recipient", "author", "name"]:
                            if field in relevant_entities and field not in extracted_entities:
                                extracted_entities[field] = entity_word
                                break
                    elif entity_label in ["ORG", "ORGANIZATION"]:
                        for field in ["company", "organization", "institution"]:
                            if field in relevant_entities and field not in extracted_entities:
                                extracted_entities[field] = entity_word
                                break
                    elif entity_label in ["LOC", "LOCATION"]:
                        for field in ["location", "address"]:
                            if field in relevant_entities and field not in extracted_entities:
                                extracted_entities[field] = entity_word
                                break
                    # Add more mappings as needed

                # 2. Regex-based extraction for common business fields
                if "date" in relevant_entities and "date" not in extracted_entities:
                    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
                    if match:
                        extracted_entities["date"] = match.group(0)
                if "total_amount" in relevant_entities and "total_amount" not in extracted_entities:
                    match = re.search(r"\$\d+(?:,\d{3})*(?:\.\d{2})?", text)
                    if match:
                        extracted_entities["total_amount"] = match.group(0)
                if "invoice_number" in relevant_entities and "invoice_number" not in extracted_entities:
                    match = re.search(r"invoice[\s#:]*(\d+)", text, re.IGNORECASE)
                    if match:
                        extracted_entities["invoice_number"] = match.group(1)
                # Add more regexes for other fields as needed

                # 3. Fallback: include all NER entities if not already mapped
                for entity in entities:
                    label = entity.get("entity_group", entity.get("entity", ""))
                    word = entity["word"].strip()
                    if word and label not in extracted_entities.values():
                        extracted_entities[label] = word

                return extracted_entities
            except Exception as e:
                logger.error(f"[EntityExtractor] Entity extraction failed: {e}")
                return {}

# Alternative approach: Global instance
_global_entity_extractor: Optional[EntityExtractor] = None
_global_lock = threading.Lock()

def get_entity_extractor(model_name: str = "dslim/bert-base-NER", use_llm: bool = False, llm_model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> EntityExtractor:
    """
    Get or create the global entity extractor instance.
    Thread-safe factory function.
    """
    global _global_entity_extractor
    
    if _global_entity_extractor is None:
        with _global_lock:
            if _global_entity_extractor is None:
                _global_entity_extractor = EntityExtractor(model_name, use_llm, llm_model_name)
    
    return _global_entity_extractor

# Django-specific initialization
class EntityExtractorManager:
    """
    Manager class for Django applications to handle model loading.
    """
    def __init__(self):
        self.extractor = None
        
    def get_extractor(self, model_name: str = "dslim/bert-base-NER", use_llm: bool = False, llm_model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> EntityExtractor:
        if self.extractor is None:
            self.extractor = EntityExtractor(model_name, use_llm, llm_model_name)
        return self.extractor

# Global manager instance
entity_manager = EntityExtractorManager()
