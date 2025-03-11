#!/usr/bin/env bash
# exit on error
set -o errexit

# Create a logs directory
mkdir -p logs

# Print Python version
python --version

# Install only CPU version of PyTorch to save space
echo "Installing PyTorch CPU only version..."
pip install torch==2.0.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install other dependencies
echo "Installing other dependencies..."
pip install --no-cache-dir $(grep -v "torch" requirements.txt)

# Run the cleanup script to reduce size
echo "Running cleanup to reduce size..."
python cleanup.py

# Remove matplotlib sample data
echo "Removing matplotlib sample data..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
MPL_DATA="${SITE_PACKAGES}/matplotlib/mpl-data/sample_data"
if [ -d "$MPL_DATA" ]; then
    rm -rf "$MPL_DATA"
    echo "Removed $MPL_DATA"
fi

# Remove scikit-learn datasets
echo "Removing scikit-learn datasets..."
SKLEARN_DATASETS="${SITE_PACKAGES}/sklearn/datasets"
if [ -d "$SKLEARN_DATASETS" ]; then
    rm -rf "$SKLEARN_DATASETS"
    echo "Removed $SKLEARN_DATASETS"
fi

# Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Print sizes of installed packages for debugging
echo "Size of site-packages directory:"
du -sh $SITE_PACKAGES

echo "Largest packages:"
pip list --format freeze | cut -d= -f1 | xargs pip show | grep -E "^Name:|^Location:|^Size:" | grep -B2 Size | grep -v "^--$" | paste -d " " - - - | sort -k5 -n -r | head -n 10

# Create a .env file from environment variables for local testing
echo "Creating .env file for local development..."
touch .env

echo "Build completed successfully!"