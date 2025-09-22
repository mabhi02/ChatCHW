@echo off
REM Setup script for ChatCHW CPU-only installation on Windows
REM This script sets up the environment for Windows users without CUDA

echo 🚀 Setting up ChatCHW for CPU-only operation...
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies (CPU-only PyTorch)...
pip install -r requirements-cpu-only.txt
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Verify PyTorch installation
echo 🧪 Verifying PyTorch installation...
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ Device: {\"CPU\" if not torch.cuda.is_available() else \"GPU\"}')"

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file template...
    (
        echo # ChatCHW Environment Variables
        echo # Replace the placeholder values with your actual API keys
        echo.
        echo # OpenAI API Key ^(required^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # Pinecone API Key ^(required^)
        echo PINECONE_API_KEY=your_pinecone_api_key_here
        echo.
        echo # Database URL ^(optional - for NeonDB^)
        echo DATABASE_URL=your_database_url_here
        echo.
        echo # Frontend URL ^(for CORS^)
        echo FRONTEND_URL=http://localhost:3000
        echo.
        echo # Backend Port ^(optional^)
        echo PORT=10000
    ) > .env
    echo ⚠️  Please edit .env file with your actual API keys
) else (
    echo ✅ .env file already exists
)

REM Test imports
echo 🔍 Testing critical imports...
python -c "import torch; import transformers; import numpy as np; import pandas as pd; import flask; import openai; import pinecone; print('✅ All critical dependencies imported successfully')"
if %errorlevel% neq 0 (
    echo ❌ Error: Failed to import some dependencies
    pause
    exit /b 1
)

echo.
echo 🎉 Setup completed successfully!
echo ==============================================
echo.
echo Next steps:
echo 1. Edit the .env file with your API keys:
echo    notepad .env
echo.
echo 2. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
echo 3. Run the backend:
echo    python app.py
echo.
echo 4. In another Command Prompt, run the frontend:
echo    cd ..\Frontend
echo    npm install
echo    npm run dev
echo.
echo 📖 For detailed instructions, see SETUP_NO_CUDA.md
echo.
echo ⚠️  Note: CPU-only mode will be slower than GPU mode,
echo    but all functionality will work correctly.
echo.
pause
