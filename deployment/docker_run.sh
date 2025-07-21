#!/bin/bash

# Build the Docker image
docker build -t health-care-doc-system -f deployment/Dockerfile .


docker run --rm -it \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/credentials/google_app_credentials.json:/app/credentials/google_app_credentials.json:ro \
  -p 8000:8000 \
  --name health-care-doc-system health-care-doc-system 