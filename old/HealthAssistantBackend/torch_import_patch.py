#!/usr/bin/env python3
"""
This file should be imported at the start of your application to patch PyTorch imports.
"""
import sys
import types
import os

def create_dummy_module(name):
    """Create a dummy module and add it to sys.modules."""
    module = types.ModuleType(name)
    module.__file__ = f"<{name} stub>"
    module.__name__ = name
    module.__path__ = []
    module.__package__ = '.'.join(name.split('.')[:-1])
    sys.modules[name] = module
    return module

def ensure_module_exists(full_name):
    """Ensure a module and all its parent modules exist."""
    parts = full_name.split('.')
    current = ''
    
    for i, part in enumerate(parts):
        if i == 0:
            current = part
        else:
            current = f"{current}.{part}"
            
        if current not in sys.modules:
            module = create_dummy_module(current)
            
            # Add as attribute to parent
            if i > 0:
                parent_name = '.'.join(parts[:i])
                if parent_name in sys.modules:
                    parent = sys.modules[parent_name]
                    setattr(parent, part, module)

# Create dummy modules for known problematic imports
modules_to_create = [
    'torch.cuda',
    'torch.distributed',
    'torch.distributed.rpc',
    'torch.distributed.autograd',
    'torch.testing',
    'torch.multiprocessing',
    'torch.random',
    'torch.storage',
    'torch.utils'
]

for module_name in modules_to_create:
    ensure_module_exists(module_name)

# Add basic functionality to CUDA module
if 'torch.cuda' in sys.modules:
    cuda_module = sys.modules['torch.cuda']
    setattr(cuda_module, 'is_available', lambda: False)
    setattr(cuda_module, 'device_count', lambda: 0)

# Add basic functionality to distributed module
if 'torch.distributed' in sys.modules:
    dist_module = sys.modules['torch.distributed']
    setattr(dist_module, 'is_available', lambda: False)
    setattr(dist_module, 'is_initialized', lambda: False)

# Install meta path hook to create missing modules on demand
class TorchImportFixer:
    def __init__(self):
        self.handled_modules = set()
    
    def find_module(self, fullname, path=None):
        if fullname.startswith('torch.') and not fullname in self.handled_modules:
            return self
        return None
    
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
            
        print(f"Auto-creating stub module: {fullname}")
        self.handled_modules.add(fullname)
        return create_dummy_module(fullname)

# Install the import fixer
sys.meta_path.insert(0, TorchImportFixer())

print("PyTorch import patches applied successfully!")