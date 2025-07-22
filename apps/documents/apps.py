from django.apps import AppConfig
import logging
from ml_pipeline.entity_extractor.extractor import get_entity_extractor
logger = logging.getLogger(__name__)


class DocumentsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.documents'

    def ready(self):
        """Initialize the entity extractor when Django starts up."""
        try:
            logger.info("Initializing Entity Extractor...")
            get_entity_extractor()
            logger.info("Entity Extractor initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Entity Extractor", exc_info=True)
