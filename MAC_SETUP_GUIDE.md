# Mac Setup Guide for ChatCHW

## Overview
This guide provides specific instructions for setting up ChatCHW on macOS systems. The application is fully compatible with Mac and will run in CPU-only mode.

## Prerequisites

### System Requirements
- **macOS**: 10.14 (Mojave) or later
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **Python**: 3.8 or later

### Check Your System
```bash
# Check macOS version
sw_vers

# Check Python version
python3 --version

# Check available memory
system_profiler SPHardwareDataType | grep "Memory:"
```

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd ChatCHW
```

2. **Run the automated setup script**:
```bash
cd HealthAssistantBackend
chmod +x setup_cpu_only.sh
./setup_cpu_only.sh
```

3. **Follow the on-screen instructions** to complete setup.

### Method 2: Manual Setup

1. **Install Python** (if not already installed):
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
# Visit https://python.org/downloads/macos/
```

2. **Create virtual environment**:
```bash
cd HealthAssistantBackend
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements-cpu-only.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env  # If example exists
# Or create manually
nano .env
```

## Mac-Specific Considerations

### Apple Silicon (M1/M2/M3) Macs

**Advantages:**
- Excellent performance for CPU-only ML workloads
- Native ARM64 support for most packages
- Better memory efficiency

**Setup Notes:**
- Use `python3` instead of `python` (if you have both Python 2 and 3)
- Some packages may take longer to install on first run
- Rosetta 2 is not needed for this setup

### Intel Macs

**Setup Notes:**
- Standard x86_64 installation
- All packages install normally
- Good performance for CPU-only workloads

### Memory Management

**For Macs with limited RAM (8GB or less):**
```bash
# Monitor memory usage
top -o MEM

# Close unnecessary applications before running
# Consider using Activity Monitor to free up memory
```

**For Macs with sufficient RAM (16GB+):**
- No special configuration needed
- Can run other applications alongside ChatCHW

## Running the Application

### Backend (Terminal 1)
```bash
cd HealthAssistantBackend
source venv/bin/activate
python app.py
```

### Frontend (Terminal 2)
```bash
cd Frontend
npm install  # First time only
npm run dev
```

### Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:10000

## Mac-Specific Troubleshooting

### Common Issues

1. **Python not found**
```bash
# Add Python to PATH (if using Homebrew)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Or create alias
echo 'alias python3=/opt/homebrew/bin/python3' >> ~/.zshrc
```

2. **Permission denied errors**
```bash
# Fix script permissions
chmod +x setup_cpu_only.sh

# Fix virtual environment permissions
chmod -R 755 venv/
```

3. **Package installation fails**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with verbose output for debugging
pip install -r requirements-cpu-only.txt -v
```

4. **Memory issues**
```bash
# Monitor memory usage
vm_stat

# Free up memory
sudo purge
```

### Performance Optimization

1. **Close unnecessary applications**
2. **Use Activity Monitor to identify memory-heavy processes**
3. **Consider using `htop` for better process monitoring**:
```bash
brew install htop
htop
```

### Development Environment

**Recommended Mac development setup:**
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install useful development tools
brew install git node python@3.11

# Install code editor (VS Code)
brew install --cask visual-studio-code

# Install terminal enhancements
brew install zsh zsh-completions
```

## Production Deployment on Mac

### Using PM2 (Process Manager)
```bash
# Install PM2 globally
npm install -g pm2

# Start backend with PM2
cd HealthAssistantBackend
source venv/bin/activate
pm2 start app.py --name "chat-chw-backend" --interpreter python

# Start frontend with PM2
cd ../Frontend
pm2 start "npm run start" --name "chat-chw-frontend"

# Save PM2 configuration
pm2 save
pm2 startup
```

### Using Docker (Alternative)
```bash
# Build and run with Docker
docker-compose up -d
```

## Mac-Specific Environment Variables

Add to your `.env` file:
```bash
# Mac-specific settings
PYTHONPATH=/opt/homebrew/lib/python3.11/site-packages
FLASK_ENV=development
FLASK_DEBUG=1

# For better performance on Apple Silicon
OMP_NUM_THREADS=8  # Adjust based on your CPU cores
```

## Testing Your Setup

### Verification Script
```bash
#!/bin/bash
echo "Testing ChatCHW setup on Mac..."

# Test Python
python3 --version

# Test virtual environment
source venv/bin/activate
python -c "import sys; print(f'Python path: {sys.executable}')"

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test other dependencies
python -c "import transformers, numpy, pandas, flask; print('✅ All dependencies working')"

# Test API endpoints (if backend is running)
curl -s http://localhost:10000/api/health || echo "Backend not running"

echo "Setup verification complete!"
```

## Performance Expectations

### Typical Performance on Mac
- **Apple Silicon (M1/M2/M3)**: Excellent performance, comparable to mid-range GPUs for CPU workloads
- **Intel Macs**: Good performance, suitable for development and moderate production use
- **Memory usage**: 2-4GB RAM during operation
- **Response times**: 2-5 seconds for ML operations (vs 0.5-1 second on GPU)

### Optimization Tips
1. **Use SSD storage** for better I/O performance
2. **Close other applications** to free up memory
3. **Use external cooling** for sustained high-load operations
4. **Monitor temperature** using apps like TG Pro or iStat Menus

## Support and Resources

### Mac-Specific Resources
- [Homebrew](https://brew.sh/) - Package manager for Mac
- [pyenv](https://github.com/pyenv/pyenv) - Python version management
- [Activity Monitor](https://support.apple.com/guide/activity-monitor/) - Built-in system monitoring

### Getting Help
- Check the main `SETUP_NO_CUDA.md` for general troubleshooting
- Review error logs in Terminal
- Use `pip install` with `-v` flag for verbose output
- Check system resources with Activity Monitor

## Summary

✅ **ChatCHW works great on Mac**
✅ **No GPU required**
✅ **Apple Silicon and Intel Macs supported**
✅ **Automated setup available**
✅ **Good performance for CPU-only workloads**

The Mac setup is straightforward and the application will run smoothly on any modern Mac without requiring any GPU hardware.
