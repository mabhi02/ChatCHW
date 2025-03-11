FROM python:3.9-slim

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with minimal extra packages
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.9/site-packages -name "__pycache__" -delete \
    && rm -rf /root/.cache/pip

# Remove test directories from installed packages to save space
RUN find /usr/local/lib/python3.9/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.9/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true

# Remove torch cuda files since we're using CPU only
RUN rm -rf /usr/local/lib/python3.9/site-packages/torch/cuda

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Run the application
CMD gunicorn app:app --bind 0.0.0.0:$PORT