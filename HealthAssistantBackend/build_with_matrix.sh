#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting deployment build with MATRIX support..."

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create our comprehensive PyTorch fix script
cat > complete_torch_fix.py << 'EOF'
#!/usr/bin/env python3
import os
import site
import sys
import importlib.util
import shutil
import re

def create_module_with_submodules(base_path, module_path):
    """Create a module and all its parent modules."""
    parts = module_path.split('/')
    current_path = base_path
    
    # Create each directory in the path
    for part in parts:
        current_path = os.path.join(current_path, part)
        os.makedirs(current_path, exist_ok=True)
        
        # Create or ensure __init__.py exists
        init_file = os.path.join(current_path, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# Stub module for {part}\n")
            print(f"Created stub module: {init_file}")

def setup_torch_stubs():
    """Create all necessary PyTorch stub modules and fix imports."""
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    torch_path = os.path.join(site_packages, 'torch')
    
    if not os.path.exists(torch_path):
        print(f"Error: PyTorch not found at {torch_path}")
        return False
    
    # Essential modules that PyTorch tries to import internally
    required_modules = [
        'cuda',
        'distributed',
        'distributed/rpc',
        'distributed/autograd',
        'distributed/optim',
        'futures',
        'testing',
        'nvtx'
    ]
    
    # Create all required modules
    for module in required_modules:
        create_module_with_submodules(torch_path, module)
    
    # Add specific functionality to cuda module
    cuda_init = os.path.join(torch_path, 'cuda', '__init__.py')
    with open(cuda_init, 'w') as f:
        f.write("""# Stub module for CUDA
def is_available():
    return False

def device_count():
    return 0

def get_device_name(device=None):
    return "CPU"

def current_device():
    return 0

class _CudaDeviceProperties:
    def __init__(self, device):
        self.name = "CPU"
        self.major = 0
        self.minor = 0
        self.total_memory = 0
        
def get_device_properties(device):
    return _CudaDeviceProperties(device)

# Add any other necessary CUDA functions with dummy implementations
""")
    
    # Add functionality to distributed module
    dist_init = os.path.join(torch_path, 'distributed', '__init__.py')
    with open(dist_init, 'w') as f:
        f.write("""# Stub module for distributed
def is_available():
    return False

def is_initialized():
    return False

def get_rank():
    return 0

def get_world_size():
    return 1

# Add any other needed distributed functions
""")
    
    # Add functionality to rpc module
    rpc_init = os.path.join(torch_path, 'distributed', 'rpc', '__init__.py')
    with open(rpc_init, 'w') as f:
        f.write("""# Stub module for distributed.rpc
def is_available():
    return False

def init_rpc(*args, **kwargs):
    return None

def shutdown(*args, **kwargs):
    return None

# Define required classes and functions
class WorkerInfo:
    def __init__(self):
        self.id = 0
        self.name = "worker0"

def get_worker_info():
    return WorkerInfo()
""")
    
    # Patch critical files that have hard-coded imports
    files_to_patch = [
        '_jit_internal.py',
        'distributed/__init__.py',
        'nn/parallel/distributed.py',
        'multiprocessing/reductions.py',
        'functional.py',
        'serialization.py'
    ]
    
    for file_path in files_to_patch:
        full_path = os.path.join(torch_path, file_path)
        if os.path.exists(full_path):
            patch_file_imports(full_path)
    
    return True

def patch_file_imports(file_path):
    """Patch a file to handle imports safely with try-except blocks."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace imports that might fail with try-except blocks
        modified = False
        
        # Pattern for standard imports
        imports = re.findall(r'import torch\.[a-zA-Z0-9_.]+', content)
        for imp in imports:
            # Skip already wrapped imports
            if f"try:\n    {imp}" in content:
                continue
                
            # Create safe import
            safe_import = f"try:\n    {imp}\nexcept ImportError:\n    pass"
            content = content.replace(imp, safe_import)
            modified = True
        
        # Pattern for from imports
        from_imports = re.findall(r'from torch\.[a-zA-Z0-9_.]+\s+import', content)
        for imp in from_imports:
            # Skip already wrapped imports
            if f"try:\n    {imp}" in content:
                continue
                
            # Find the full import line (this is approximate)
            import_pattern = f"{imp}.*?($|\n)"
            matches = re.findall(f"{re.escape(imp)}.*?($|\n)", content, re.DOTALL)
            
            if matches:
                orig_import = matches[0]
                # Create safe import
                safe_import = f"try:\n    {orig_import.strip()}\nexcept ImportError:\n    pass\n"
                content = content.replace(orig_import, safe_import)
                modified = True
        
        if modified:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Patched imports in {file_path}")
    
    except Exception as e:
        print(f"Error patching {file_path}: {e}")

def modify_cleanup_script():
    """Modify the cleanup script to preserve our stubs."""
    cleanup_path = 'cleanup.py'
    if not os.path.exists(cleanup_path):
        print(f"Warning: {cleanup_path} not found. Skipping modification.")
        return
    
    with open(cleanup_path, 'r') as f:
        content = f.read()
    
    # Modify the dirs_to_remove to avoid removing our stub modules
    for module in ['cuda', 'distributed', 'testing']:
        pattern = f"['\"]({module})['\"]"
        content = re.sub(pattern, f"# '{module}'", content)
    
    with open(cleanup_path, 'w') as f:
        f.write(content)
    
    print(f"Modified {cleanup_path} to preserve stub modules")

if __name__ == '__main__':
    print("Setting up PyTorch stubs and fixing imports...")
    if setup_torch_stubs():
        modify_cleanup_script()
        # Verify PyTorch can be imported
        try:
            import torch
            import torch.nn as nn
            print(f"Success! PyTorch {torch.__version__} imported correctly.")
        except Exception as e:
            print(f"Error importing PyTorch after fixes: {e}")
    else:
        print("Failed to set up PyTorch stubs.")
EOF

# Apply PyTorch fixes - do this BEFORE any cleanup
echo "Setting up PyTorch stub modules and fixing imports..."
python complete_torch_fix.py

# Fix application imports
echo "Fixing application imports..."
python fix_imports.py

# Test if torch imports work
echo "Testing PyTorch imports..."
python -c "import torch; import torch.nn as nn; print('PyTorch imports successful!')"

# Now run the cleanup script (which has been modified by our fix script)
echo "Running cleanup script..."
python cleanup.py

# Re-test PyTorch imports after cleanup
echo "Testing PyTorch imports after cleanup..."
python -c "import torch; import torch.nn as nn; print('PyTorch still works after cleanup!')"

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"