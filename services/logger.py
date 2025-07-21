import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

logger = logging.getLogger('healthcare_app')
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers in interactive environments
if not logger.handlers:
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Optional: also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

# Usage example:
# from services.logger import logger
# logger.info('This is an info message') 