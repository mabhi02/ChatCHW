# Running ChatCHW Without CUDA (CPU-Only Setup)

## Overview
This project is already configured to run without CUDA/GPU support. All ML operations will use CPU instead of GPU, which is perfect for Mac users and systems without dedicated graphics cards.

## Quick Start

### Backend (Python/Flask)
```bash
# Navigate to backend directory
cd HealthAssistantBackend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies (CPU-only PyTorch)
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Copy and edit with your API keys

# Run the backend
python app.py
```

### Frontend (Next.js)
```bash
# Navigate to frontend directory
cd Frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## Environment Variables Needed

Create a `.env` file in `HealthAssistantBackend/` with:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
DATABASE_URL=your_database_url_here  # Optional: for NeonDB
FRONTEND_URL=http://localhost:3000  # For local development
```

## Platform-Specific Instructions

### Mac Users
The setup is identical to other platforms since we're using CPU-only PyTorch:

```bash
# Make sure you have Python 3.8+ installed
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# Run the application
python3 app.py
```

### Windows Users (No CUDA)
```bash
# Use Command Prompt or PowerShell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Linux Users
```bash
# Install Python 3.8+ if not already installed
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Performance Notes

### CPU-Only Performance
- **Slower ML operations**: CPU processing is slower than GPU for ML tasks
- **Memory usage**: CPU-only PyTorch uses more RAM but less VRAM
- **Suitable for**: Development, testing, and production with moderate load

### Recommended System Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Storage**: 2GB free space for dependencies

## Troubleshooting

### Common Issues

1. **PyTorch Import Errors**
   - The project includes `torch_import_patch.py` to handle CUDA-related import errors
   - If you encounter issues, ensure this file is imported before other torch modules

2. **Memory Issues**
   - CPU-only PyTorch can use significant RAM
   - Close other applications if you encounter out-of-memory errors
   - Consider reducing batch sizes in the code if needed

3. **Slow Performance**
   - This is expected with CPU-only processing
   - Consider using smaller models or reducing complexity for better performance

### Verification Commands

Test that everything is working:
```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test other dependencies
python -c "import transformers, numpy, pandas; print('All dependencies loaded successfully')"
```

## Alternative Setup Scripts

### Automated Setup Script (Mac/Linux)
```bash
#!/bin/bash
# setup_cpu_only.sh

echo "Setting up ChatCHW for CPU-only operation..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
DATABASE_URL=your_database_url_here
FRONTEND_URL=http://localhost:3000
EOF
    echo "Please edit .env file with your actual API keys"
fi

echo "Setup complete! Activate with: source venv/bin/activate"
echo "Then run: python app.py"
```

### Windows Setup Script
```batch
@echo off
echo Setting up ChatCHW for CPU-only operation...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file template...
    echo OPENAI_API_KEY=your_openai_api_key_here > .env
    echo PINECONE_API_KEY=your_pinecone_api_key_here >> .env
    echo DATABASE_URL=your_database_url_here >> .env
    echo FRONTEND_URL=http://localhost:3000 >> .env
    echo Please edit .env file with your actual API keys
)

echo Setup complete! Activate with: venv\Scripts\activate.bat
echo Then run: python app.py
```

## Production Deployment

### For CPU-Only Production
- Use `gunicorn` for better performance: `gunicorn -w 4 -b 0.0.0.0:10000 app:app`
- Consider using a reverse proxy like nginx
- Monitor memory usage and scale horizontally if needed

### Cloud Deployment Options
- **Render**: Already configured in the project
- **Heroku**: Add Procfile with `web: gunicorn app:app`
- **DigitalOcean App Platform**: CPU-only deployment supported
- **AWS EC2**: Any instance type will work

## Summary

✅ **You can absolutely run this without CUDA**
✅ **Mac users can use the same setup as everyone else**
✅ **Performance will be slower but functional**
✅ **All features work the same way**

The project is well-designed for CPU-only operation and should work seamlessly on any platform without GPU requirements.
