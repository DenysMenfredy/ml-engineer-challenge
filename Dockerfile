# Use official Python image as base
FROM python:3.10-slim

# Set environment variables early for better layer caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/google_app_credentials.json

# Set work directory
WORKDIR /app

# Install system dependencies in one layer, clean up properly
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        tesseract-ocr \
        poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better caching
COPY pyproject.toml uv.lock ./
RUN pip install --upgrade pip && \
    pip install uv && \
    uv sync --all-groups && \
    uv add gunicorn

# Copy only the necessary project files (use .dockerignore to exclude unnecessary files)
COPY . .

# Expose port (if running Django server)
EXPOSE 8000

# Use exec form for CMD, and use uvicorn/gunicorn for production if possible
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]
