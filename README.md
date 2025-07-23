# Health Care Document Processing System

A comprehensive document processing system that uses OCR (Optical Character Recognition) to extract text from business documents, classifies them, extracts relevant entities (using NER, Hugging Face LLMs, or Ollama), and stores them in a vector database (ChromaDB) for efficient retrieval and analysis.

## Features

- **Multi-format OCR**: Support for various image formats (JPG, PNG, TIFF, PDF, etc.)
- **Google Cloud Vision API** and **Tesseract OCR**: High-accuracy text extraction
- **Image Preprocessing**: Noise removal, skew correction, binarization
- **Vector Database**: ChromaDB integration for semantic search and storage of documents, types, and extracted entities
- **Django Integration**: Web framework for API and management
- **Document Classification**: Automatic classification using vector similarity and metadata
- **Entity Extraction**: Supports standard NER, Hugging Face LLMs, and local LLMs via Ollama (prompt-based extraction)
- **Configurable Pipeline**: Easily switch between entity extraction methods
- **Robust Logging**: Uses a rotating file and console logger.

## Project Structure

```
health-care-document-processing-system/
├── data/
│   └── docs-sm/                    # Document dataset
│       ├── invoice/                # Document class example
│       ├── form/                   # Document class example
│       └── ...                     # More document classes
├── apps/                           # Django apps (documents, processing)
├── config/                         # Django settings
├── ml_pipeline/                    # ML processing pipeline
│   ├── ocr/                        # OCR modules
│   ├── dataset/                    # Dataset generation
│   ├── entity_extractor/           # Entity extraction (NER, LLM, Ollama)
│   └── ...
├── services/                       # Logger, etc.
├── main.py                         # Main execution script
├── manage.py                       # Django management script
├── requirements.txt                # Project dependencies (generated from pyproject.toml)
└── README.md                       # This file
```

## Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** (for local OCR processing)
3. **Google Cloud Vision API** credentials (for cloud OCR)
4. **Ollama** (optional, for local LLM entity extraction)

### Installing Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Google Cloud Vision API Setup

1. Create a Google Cloud project
2. Enable the Cloud Vision API
3. Create a service account and download the JSON key file
4. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

### Ollama Setup (Optional)

1. [Install Ollama](https://ollama.com/download) on your machine
2. Pull and run your desired model (e.g., gemma3:1b):
   ```bash
   ollama pull gemma3:1b
   ollama run gemma3:1b
   ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd health-care-document-processing-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or using uv (if available)
   uv sync
   ```

3. **Verify setup:**
   ```bash
   python test_setup.py
   ```

## Document Processing Pipeline

1. **OCR Extraction:** Extracts text from documents using Tesseract or Google Vision.
2. **Text Cleaning:** Cleans and normalizes text for ML processing.
3. **Document Classification:** Identifies document type using vector similarity search in ChromaDB.
4. **Entity Extraction:**
   - **NER:** Standard named entity recognition (default)
   - **Hugging Face LLM:** Prompt-based extraction using instruction-tuned models
   - **Ollama:** Local LLM extraction via REST API (e.g., gemma3:1b, mistral, llama2)
5. **Storage:** Stores cleaned text, document type, and extracted entities (as JSON) in ChromaDB for semantic search and retrieval.

## Usage

### Batch Processing (Django Management Command)

Process a dataset of documents:
```bash
python manage.py process_documents --input_dir data/docs-sm
```

### API Usage

Start the Django development server:
```bash
python manage.py runserver
```

Upload a document via the API:
- Endpoint: `POST /api/documents/`
- Form field: `file` (the document to upload)

#### Example API Response
```json
{
  "filename": "invoice123.pdf",
  "document_type": "invoice",
  "text": "invoice from acmecorp dated 2023-01-01 for $1000 ...",
  "entities": {
    "invoice_number": "123",
    "date": "2023-01-01",
    "total_amount": "$1000",
    "customer_name": "AcmeCorp"
  },
  "confidence": 0.98
}
```

### Entity Extraction Configuration

You can configure the entity extraction method in your code:
```python
from ml_pipeline.entity_extractor.extractor import EntityExtractor

# Use Ollama (local LLM)
extractor = EntityExtractor(use_ollama=True, ollama_model="gemma3:1b")

# Use Hugging Face LLM
extractor = EntityExtractor(use_llm=True, llm_model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")

# Use standard NER/regex (default)
extractor = EntityExtractor()
```

## Docker Compose Usage

You can build and run the application using Docker Compose:

1. **Build the Docker image:**
   ```bash
   docker compose build --no-cache
   ```

2. **Start the application:**
   ```bash
   docker compose up
   ```

- The app will be available at http://localhost:8080
- Logs will be written to the ./logs directory on your host machine.
- Make sure your Google Cloud credentials are available at ./credentials/google_app_credentials.json

## Development

### Running Tests

```bash
python -m unittest discover
```

### Adding New Document Types or Entities
- Add new folder(s) to your dataset for new document types
- Update `entity_mapping` in `ml_pipeline/entity_extractor/extractor.py` to include new fields
- Add or adjust regexes or prompt instructions as needed

### Logging
- Logs are written to `logs/app.log` and the console
- Logger name: `document_processing`
- Log level can be set in `services/logger.py`

## Requirements & Testing
- All dependencies are listed in `requirements.txt` (generated from `pyproject.toml`)
- Unit tests for entity extraction and dataset generation are included in `tests/` and `ml_pipeline/entity_extractor/`

## Troubleshooting

- **Timeouts:** Increase Gunicorn or reverse proxy timeout if LLM extraction is slow
- **Ollama errors:** Ensure Ollama is running and the model is pulled
- **ChromaDB errors:** Ensure metadata values are JSON-serializable (e.g., entities as JSON string)
- **OCR errors:** Check Tesseract/Google Vision installation and credentials


## License
This project is licensed under the MIT License - see the LICENSE file for details.

