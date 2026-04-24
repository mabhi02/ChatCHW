#!/usr/bin/env python3
import os
import re

def fix_imports_in_file(file_path):
    """Fix import statements in Python files."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add torch.cuda fix
    if 'import torch' in content and 'import torch.nn' in content:
        # Ensure torch import has cuda handling
        torch_import_fix = '''
try:
    import torch
except ImportError as e:
    if "torch.cuda" in str(e):
        # If the error is related to torch.cuda, we need a custom import approach
        import importlib.util
        import sys
        
        # Create a fake torch.cuda module to prevent import errors
        class DummyCuda:
            is_available = lambda: False
            
        # Import torch without cuda
        spec = importlib.util.find_spec("torch")
        torch = importlib.util.module_from_spec(spec)
        sys.modules["torch"] = torch
        
        # Add the dummy cuda module
        sys.modules["torch.cuda"] = DummyCuda()
        
        # Continue with torch import
        spec.loader.exec_module(torch)
    else:
        # If it's a different error, raise it
        raise e
'''
        # Only add if not already there
        if 'try:\n    import torch' not in content:
            content = re.sub(r'import torch', torch_import_fix, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Fix imports in all Python files in the project."""
    print("Starting to fix imports...")
    
    # Fix imports in app.py and chad.py
    for file_name in ['app.py', 'chad.py']:
        if os.path.exists(file_name):
            fix_imports_in_file(file_name)
    
    print("Done!")

if __name__ == "__main__":
    main()