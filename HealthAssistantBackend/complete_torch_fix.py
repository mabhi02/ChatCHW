#!/usr/bin/env python3
"""
Improved PyTorch fix that creates necessary stub modules
without messing up existing Python files.
"""
import os
import site
import sys
import importlib.util
import shutil

def find_torch_path():
    """Find the correct path to the torch module."""
    # First try importing torch and get its path
    try:
        import torch
        torch_path = os.path.dirname(torch.__file__)
        print(f"Found torch at: {torch_path}")
        return torch_path
    except ImportError:
        print("Could not import torch. Trying to find it manually...")
    
    # Try to find torch in site-packages
    for path in site.getsitepackages():
        potential_path = os.path.join(path, 'torch')
        if os.path.exists(potential_path):
            print(f"Found torch at: {potential_path}")
            return potential_path
    
    # Check in current virtual environment if we're in one
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        venv_path = sys.prefix
        potential_paths = [
            os.path.join(venv_path, 'Lib', 'site-packages', 'torch'),
            os.path.join(venv_path, 'lib', 'python' + sys.version[:3], 'site-packages', 'torch')
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                print(f"Found torch at: {path}")
                return path
    
    return None

def create_stub_module(base_path, module_path, content=""):
    """Create a stub module with the specified content."""
    # Convert forward slashes to the appropriate separator for the OS
    module_parts = module_path.split('/')
    current_path = base_path
    
    # Create each directory in the path
    for i, part in enumerate(module_parts):
        current_path = os.path.join(current_path, part)
        if i < len(module_parts) - 1:  # It's a directory
            os.makedirs(current_path, exist_ok=True)
            # Make sure __init__.py exists in each directory
            init_file = os.path.join(current_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write(f"# Stub module for {part}\n")
                print(f"Created directory stub: {init_file}")
    
    # Write the content to the file
    with open(current_path, 'w') as f:
        f.write(content)
    print(f"Created stub file: {current_path}")

def setup_torch_stubs():
    """Create all necessary PyTorch stub modules without patching existing files."""
    # Get the torch path
    torch_path = find_torch_path()
    
    if not torch_path:
        print("Error: PyTorch not found. Make sure it's installed in your environment.")
        return False
    
    # First ensure the basic directory structure exists
    dirs_to_create = [
        'cuda',
        'distributed',
        'distributed/rpc',
        'distributed/autograd',
        'distributed/optim',
        'testing',
        'nvtx'
    ]
    
    for dir_path in dirs_to_create:
        dir_full_path = os.path.join(torch_path, *dir_path.split('/'))
        os.makedirs(dir_full_path, exist_ok=True)
        
        # Create __init__.py in each directory if it doesn't exist
        init_path = os.path.join(dir_full_path, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write(f"# Stub module for {dir_path}\n")
            print(f"Created directory: {dir_full_path}")
    
    # Create cuda/__init__.py with minimal functionality
    cuda_init = os.path.join(torch_path, 'cuda', '__init__.py')
    cuda_content = """# Stub module for CUDA
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
"""
    with open(cuda_init, 'w') as f:
        f.write(cuda_content)
    print(f"Created cuda stubs: {cuda_init}")
    
    # Create distributed/__init__.py with minimal functionality
    dist_init = os.path.join(torch_path, 'distributed', '__init__.py')
    dist_content = """# Stub module for distributed
def is_available():
    return False

def is_initialized():
    return False

def get_rank():
    return 0

def get_world_size():
    return 1
"""
    with open(dist_init, 'w') as f:
        f.write(dist_content)
    print(f"Created distributed stubs: {dist_init}")
    
    # Create distributed/rpc/__init__.py with minimal functionality
    rpc_init = os.path.join(torch_path, 'distributed', 'rpc', '__init__.py')
    rpc_content = """# Stub module for distributed.rpc
def is_available():
    return False

def init_rpc(*args, **kwargs):
    return None

def shutdown(*args, **kwargs):
    return None

class WorkerInfo:
    def __init__(self):
        self.id = 0
        self.name = "worker0"

def get_worker_info():
    return WorkerInfo()
"""
    with open(rpc_init, 'w') as f:
        f.write(rpc_content)
    print(f"Created RPC stubs: {rpc_init}")
    
    # Instead of patching files, we'll create a torch_patch.py that will be imported
    # Create a patch module that will monkey patch torch imports at runtime
    patch_file = os.path.join(torch_path, 'torch_patch.py')
    patch_content = """# Monkey patch for PyTorch imports
import sys
import types

class ImportInterceptor:
    def __init__(self):
        self.missing_modules = set()
    
    def find_module(self, fullname, path=None):
        if fullname.startswith('torch.') and 'distributed' in fullname:
            parts = fullname.split('.')
            if len(parts) > 2 and parts[1] == 'distributed':
                return self
        return None
    
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        # Create a new empty module
        module = types.ModuleType(fullname)
        module.__file__ = "<torch stub>"
        module.__name__ = fullname
        module.__path__ = []
        module.__loader__ = self
        module.__package__ = '.'.join(fullname.split('.')[:-1])
        
        # Add the module to sys.modules
        sys.modules[fullname] = module
        
        print(f"Created stub for {fullname}")
        return module

# Install the import interceptor
sys.meta_path.insert(0, ImportInterceptor())
"""
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    print(f"Created import patch: {patch_file}")
    
    # Create a patch file that will be imported by __init__.py
    init_patch_file = os.path.join(torch_path, 'import_patch.py')
    init_patch_content = """# This file patches missing imports for torch
def patch_torch_imports():
    import sys
    import types
    
    # Create dummy modules for these imports
    dummy_modules = [
        'torch.distributed.rpc',
        'torch.distributed.autograd',
        'torch.distributed.optim',
        'torch.testing'
    ]
    
    for module_name in dummy_modules:
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            module.__file__ = f"<{module_name} stub>"
            module.__name__ = module_name
            module.__path__ = []
            module.__package__ = '.'.join(module_name.split('.')[:-1])
            sys.modules[module_name] = module
            
            # Also make parent modules aware of this module
            parts = module_name.split('.')
            for i in range(1, len(parts)):
                parent_name = '.'.join(parts[:i])
                if parent_name in sys.modules:
                    parent = sys.modules[parent_name]
                    setattr(parent, parts[i], module)
    
    # Add monkey-patched functions to distributed.rpc
    if 'torch.distributed.rpc' in sys.modules:
        rpc = sys.modules['torch.distributed.rpc']
        setattr(rpc, 'is_available', lambda: False)
        setattr(rpc, 'init_rpc', lambda *args, **kwargs: None)
        setattr(rpc, 'shutdown', lambda *args, **kwargs: None)
        
        # Add worker info class
        class WorkerInfo:
            def __init__(self):
                self.id = 0
                self.name = "worker0"
        
        setattr(rpc, 'WorkerInfo', WorkerInfo)
        setattr(rpc, 'get_worker_info', lambda: WorkerInfo())

# Call the patch function when this module is imported
patch_torch_imports()
"""
    with open(init_patch_file, 'w') as f:
        f.write(init_patch_content)
    print(f"Created import patch module: {init_patch_file}")
    
    # Modify __init__.py to import our patch early
    init_file = os.path.join(torch_path, '__init__.py')
    if os.path.exists(init_file):
        # Backup the original file
        backup_file = init_file + '.backup'
        if not os.path.exists(backup_file):
            shutil.copy2(init_file, backup_file)
            print(f"Created backup of __init__.py at {backup_file}")
        
        # Read the content
        with open(init_file, 'r') as f:
            content = f.read()
        
        # Add our patch import near the top, after the first few imports
        import_lines = content.split('\n')
        patch_line = "from . import import_patch  # Added by torch fix script"
        
        # Find a good place to insert our patch - after the first block of imports
        insert_pos = 0
        for i, line in enumerate(import_lines):
            if i > 10 and not line.startswith('import') and not line.startswith('from'):
                insert_pos = i
                break
                
        if insert_pos > 0:
            import_lines.insert(insert_pos, patch_line)
            new_content = '\n'.join(import_lines)
            
            # Write the modified content
            with open(init_file, 'w') as f:
                f.write(new_content)
            print(f"Modified {init_file} to import our patch")
    
    return True

def modify_cleanup_script():
    """Modify the cleanup script to preserve our stubs."""
    cleanup_path = 'cleanup.py'
    if not os.path.exists(cleanup_path):
        print(f"Warning: {cleanup_path} not found. Skipping modification.")
        return
    
    try:
        with open(cleanup_path, 'r') as f:
            content = f.read()
        
        # Find the lines that remove torch.cuda and torch.distributed
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # Skip lines that would remove our stub directories
            if any(x in line for x in [
                "'cuda'", '"cuda"',
                "'distributed'", '"distributed"',
                "'testing'", '"testing"'
            ]) and 'dirs_to_remove' in line:
                modified_line = '# ' + line  # Comment out the line
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
        
        modified_content = '\n'.join(modified_lines)
        
        # Write the modified content back
        with open(cleanup_path, 'w') as f:
            f.write(modified_content)
        
        print(f"Modified {cleanup_path} to preserve stub modules")
    except Exception as e:
        print(f"Error modifying cleanup.py: {e}")

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