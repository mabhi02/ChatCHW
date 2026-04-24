#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting build with PyTorch fixes for MATRIX..."

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Apply PyTorch fixes - do this BEFORE any cleanup
echo "Setting up PyTorch stub modules and fixing imports..."
python complete_torch_fix.py

# Fix application imports
echo "Fixing imports in app.py and chad.py..."
python fix_imports.py

# Test if torch imports work
echo "Testing PyTorch imports..."
python -c "import torch; import torch.nn as nn; print('PyTorch imports successful!')"

# Now run the cleanup script (which has been modified by our fix script)
echo "Running cleanup script..."
python cleanup.py

# Verify torch stubs are still intact after cleanup
echo "Verifying PyTorch still works after cleanup..."
python -c "import torch; import torch.nn as nn; print('PyTorch still works after cleanup!')"

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"