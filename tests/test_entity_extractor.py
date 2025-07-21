import unittest
from unittest.mock import patch
from ml_pipeline.entity_extractor.extractor import EntityExtractor

class TestEntityExtractor(unittest.TestCase):
    def test_initialization(self):
        extractor = EntityExtractor()
        self.assertIsNotNone(extractor.ner_pipeline)
        self.assertIn("invoice", extractor.entity_mapping)

    @patch("ml_pipeline.entity_extractor.extractor.pipeline")
    def test_extract_entities_basic(self, mock_pipeline):
        # Mock the Hugging Face pipeline output
        mock_pipeline.return_value = lambda text: [
            {"entity": "ORG", "word": "AcmeCorp"},
            {"entity": "DATE", "word": "2023-01-01"},
            {"entity": "MONEY", "word": "$1000"}
        ]
        extractor = EntityExtractor()
        result = extractor.extract_entities("Invoice from AcmeCorp dated 2023-01-01 for $1000.", "invoice")
        self.assertIsInstance(result, dict)

    @patch("ml_pipeline.entity_extractor.extractor.pipeline")
    def test_entity_mapping_filtering(self, mock_pipeline):
        # Simulate NER output with extra irrelevant entities
        mock_pipeline.return_value = lambda text: [
            {"entity": "ORG", "word": "AcmeCorp"},
            {"entity": "DATE", "word": "2023-01-01"},
            {"entity": "MONEY", "word": "$1000"},
            {"entity": "PERSON", "word": "John Doe"}
        ]
        extractor = EntityExtractor()
        result = extractor.extract_entities("Invoice from AcmeCorp dated 2023-01-01 for $1000.", "invoice")
        self.assertIsInstance(result, dict)

if __name__ == "__main__":
    unittest.main() 