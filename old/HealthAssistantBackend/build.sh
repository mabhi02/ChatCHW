#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting deployment build..."

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Apply PyTorch fixes BEFORE running cleanup
echo "Setting up PyTorch stub modules..."
python complete_torch_fix.py

# Fix imports for app.py and chad.py
echo "Fixing imports in app.py and chad.py..."
python fix_imports.py

# Verify PyTorch imports work before cleanup
echo "Verifying PyTorch imports before cleanup..."
python -c "import torch; import torch.nn as nn; import torch.testing; print('PyTorch imports successful!')"

# Now run the cleanup script (which has been modified to preserve necessary modules)
echo "Running cleanup script..."
python cleanup.py

# Verify PyTorch still works after cleanup
echo "Verifying PyTorch imports after cleanup..."
python -c "import torch; import torch.nn as nn; import torch.testing; print('PyTorch still works after cleanup!')"

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"