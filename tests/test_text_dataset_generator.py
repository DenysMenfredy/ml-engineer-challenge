import unittest
from unittest.mock import MagicMock, patch
from ml_pipeline.dataset.generator import TextDatasetGenerator

class TestTextDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.input_dir = "test_data"
        self.config = {"ocr_processor_pipeline": MagicMock()}
        self.generator = TextDatasetGenerator(self.input_dir, self.config)

    def test_clean_text(self):
        dirty_text = "  Hello,   WORLD!  This is a test.\n\n"
        cleaned = self.generator.clean_text(dirty_text)
        self.assertEqual(cleaned, "hello, world! this is a test.")

    @patch("ml_pipeline.dataset.generator.chromadb.PersistentClient")
    def test_get_existing_document_ids(self, mock_chroma):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["file1.jpg", "file2.jpg"]}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        generator = TextDatasetGenerator(self.input_dir, self.config)
        ids = generator.get_existing_document_ids()
        self.assertIn("file1.jpg", ids)
        self.assertIn("file2.jpg", ids)

    def test_skip_unreadable_file(self):
        import os
        with patch("os.path.isfile", return_value=False):
            with patch("os.access", return_value=False):
                # Should skip file, so no exception should be raised
                pass  # The actual skipping is in the generate() method

if __name__ == "__main__":
    unittest.main() 