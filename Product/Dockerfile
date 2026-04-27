FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# - gcc: compiling Python C extensions
# - poppler-utils: required by pdf2image for rendering PDF page regions to images
#   (we use this to crop tables/images by Unstructured's bounding boxes before
#   sending them to the vision model)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Generate Prisma client
COPY backend/prisma ./prisma
RUN prisma generate

# Copy RLM core library (skip docs, media, visualizer, tests, examples)
COPY rlm/rlm/ ./rlm/rlm/
COPY rlm/pyproject.toml ./rlm/
RUN pip install --no-cache-dir ./rlm

# Copy application code
COPY backend/ ./backend/

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["python", "-m", "backend.main"]
