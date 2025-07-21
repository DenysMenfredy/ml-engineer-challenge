# Health Care Document Processing System

A comprehensive document processing system that uses OCR (Optical Character Recognition) to extract text from healthcare documents and stores them in a vector database for efficient retrieval and analysis.

## Features

- **Multi-format OCR**: Support for various image formats (JPG, PNG, TIFF, etc.)
- **Google Cloud Vision API**: High-accuracy text extraction
- **Tesseract OCR**: Open-source OCR alternative
- **Image Preprocessing**: Noise removal, skew correction, binarization
- **Vector Database**: ChromaDB integration for semantic search
- **Django Integration**: Web framework for API and management
- **Document Classification**: Automatic classification based on folder structure

## Project Structure

```
health-care-document-processing-system/
├── data/
│   └── docs-sm/                    # Document dataset
│       ├── advertisement/          # Document class 1
│       ├── budget/                 # Document class 2
│       ├── email/                  # Document class 3
│       └── ...                     # More document classes
├── document_processing_system/     # Django project
│   ├── apps/                       # Django apps
│   ├── config/                     # Django settings
│   ├── ml_pipeline/                # ML processing pipeline
│   │   ├── ocr/                    # OCR modules
│   │   ├── dataset/                # Dataset generation
│   │   └── ...
│   ├── management/                 # Django management commands
│   └── manage.py                   # Django management script
├── main.py                         # Main execution script
├── test_setup.py                   # Setup verification script
└── pyproject.toml                  # Project dependencies
```

## Prerequisites

1. **Python 3.13+**
2. **Tesseract OCR** (for local OCR processing)
3. **Google Cloud Vision API** credentials (for cloud OCR)

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

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd health-care-document-processing-system
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Verify setup:**
   ```bash
   python test_setup.py
   ```

## Usage

### Quick Start

1. **Test the setup:**
   ```bash
   python test_setup.py
   ```

2. **Process documents:**
   ```bash
   python main.py
   ```

3. **Using Django management command:**
   ```bash
   # From project root (recommended)
   python run_django_command.py process_documents data/docs-sm
   
   # Or from Django directory
   cd document_processing_system
   PYTHONPATH=.. python manage.py process_documents ../data/docs-sm
   ```

### Configuration

The system can be configured through the `config` parameter in the `TextDatasetGenerator`:

```python
config = {
    "ocr_processor_pipeline": OCRPipeline(
        GoogleCloudVisionOCRProcessor({
            "language_hints": ["en"]
        })
    )
}
```

### Document Structure

Place your documents in the `data/docs-sm/` directory with the following structure:

```
data/docs-sm/
├── class1/          # Documents of class 1
│   ├── doc1.jpg
│   ├── doc2.png
│   └── ...
├── class2/          # Documents of class 2
│   ├── doc3.jpg
│   └── ...
└── ...
```

The folder names will be used as class labels for document classification.

## API Usage

### Django REST API

Start the Django development server:
```bash
cd document_processing_system
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/`

### Vector Database Queries

The processed documents are stored in ChromaDB for semantic search:

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
collection = client.get_collection("documents")

# Search for similar documents
results = collection.query(
    query_texts=["medical report"],
    n_results=5
)
```

## Development

### Running Tests

```bash
python test_setup.py
```

### Adding New OCR Processors

1. Create a new processor class inheriting from `BaseOCRProcessor`
2. Implement the required methods
3. Update the pipeline configuration

### Adding New Preprocessing Steps

1. Add new methods to the `OCRPreprocessor` class
2. Update the `preprocess_pipeline` method
3. Configure the pipeline to use the new preprocessing

## Troubleshooting

### Common Issues

1. **Tesseract not found:**
   - Install Tesseract OCR
   - Ensure it's in your system PATH

2. **Google Cloud Vision API errors:**
   - Check your service account credentials
   - Verify the API is enabled
   - Check your quota limits

3. **Import errors:**
   - Run `python test_setup.py` to verify the setup
   - Check that all dependencies are installed

4. **Memory issues with large datasets:**
   - Process documents in smaller batches
   - Use the Django management command for better memory management

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export DEBUG=1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the test output for specific error messages
