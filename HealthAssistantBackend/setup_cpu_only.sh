#!/bin/bash
# Setup script for ChatCHW CPU-only installation
# Compatible with Mac, Linux, and WSL

set -e  # Exit on any error

echo "🚀 Setting up ChatCHW for CPU-only operation..."
echo "=============================================="

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required. Current version: $python_version"
    echo "Please install Python 3.8 or higher from https://python.org"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies (CPU-only PyTorch)..."
pip install -r requirements-cpu-only.txt

# Verify PyTorch installation
echo "🧪 Verifying PyTorch installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Device: {\"CPU\" if not torch.cuda.is_available() else \"GPU\"}')
"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file template..."
    cat > .env << 'EOF'
# ChatCHW Environment Variables
# Replace the placeholder values with your actual API keys

# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key (required)
PINECONE_API_KEY=your_pinecone_api_key_here

# Database URL (optional - for NeonDB)
DATABASE_URL=your_database_url_here

# Frontend URL (for CORS)
FRONTEND_URL=http://localhost:3000

# Backend Port (optional)
PORT=10000
EOF
    echo "⚠️  Please edit .env file with your actual API keys"
else
    echo "✅ .env file already exists"
fi

# Test imports
echo "🔍 Testing critical imports..."
python -c "
try:
    import torch
    import transformers
    import numpy as np
    import pandas as pd
    import flask
    import openai
    import pinecone
    print('✅ All critical dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the backend:"
echo "   python app.py"
echo ""
echo "4. In another terminal, run the frontend:"
echo "   cd ../Frontend"
echo "   npm install"
echo "   npm run dev"
echo ""
echo "📖 For detailed instructions, see SETUP_NO_CUDA.md"
echo ""
echo "⚠️  Note: CPU-only mode will be slower than GPU mode,"
echo "   but all functionality will work correctly."
