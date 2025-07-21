# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Set Google Application Credentials environment variable (update the path as needed)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/google_app_credentials.json

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --upgrade pip && \
    pip install uv && \
    uv sync --all-groups

# Copy project, but exclude data directory via .dockerignore
COPY . .

# Expose port (if running Django server)
EXPOSE 8000

# Default command (can be overridden)
CMD ["uv", "run", "manage.py", "runserver", "0.0.0.0:8000"]