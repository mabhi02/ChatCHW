#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Fix Pinecone imports
python fix_imports.py

# Ensure torch.cuda exists with minimal stubs
mkdir -p $(python -c "import site; print(site.getsitepackages()[0])")/torch/cuda
echo "def is_available():
    return False

def device_count():
    return 0" > $(python -c "import site; print(site.getsitepackages()[0])")/torch/cuda/__init__.py

# Run cleanup script to reduce size
python cleanup.py

# Double-check that torch.cuda exists
if [ ! -f $(python -c "import site; print(site.getsitepackages()[0])")/torch/cuda/__init__.py ]; then
    echo "Creating missing torch.cuda/__init__.py"
    mkdir -p $(python -c "import site; print(site.getsitepackages()[0])")/torch/cuda
    echo "def is_available():
    return False

def device_count():
    return 0" > $(python -c "import site; print(site.getsitepackages()[0])")/torch/cuda/__init__.py
fi

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"