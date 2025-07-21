# Use official Python image as base
FROM python:3.10-slim

# Set environment variables early for better layer caching
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/google_app_credentials.json \
    TRANSFORMERS_NO_CUDA=1 \
    CUDA_VISIBLE_DEVICES=""

# Set work directory
WORKDIR /app

# Install system dependencies in one layer, clean up properly
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better caching
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install . && \
    pip install gunicorn

# Copy only the necessary project files (use .dockerignore to exclude unnecessary files)
COPY . .

# Expose port (if running Django server)
EXPOSE 8080

# Use exec form for CMD, and use uvicorn/gunicorn for production if possible
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8080"]
