import re

def clean_text(text: str) -> str:
        """Clean text for ML model input."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text