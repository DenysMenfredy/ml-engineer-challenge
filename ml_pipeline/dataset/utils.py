import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Clean text for ML model input.
    Steps:
    - Normalize unicode (NFKC)
    - Remove control characters
    - Remove extra whitespace
    - Convert to lowercase
    - Remove non-alphanumeric and non-basic punctuation (.,!?)
    - Remove accents from characters
    """
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Remove accents
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    # Remove control characters
    text = re.sub(r'[\r\n\t\x0b\x0c]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric and non-basic punctuation (.,!?)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text