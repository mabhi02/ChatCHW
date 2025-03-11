#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Fix Pinecone imports
python fix_imports.py

# Run cleanup script to reduce size
python cleanup.py

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"