#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Starting deployment build..."

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Create the simple stub creator script
cat > create_torch_stubs.py << 'EOF'
#!/usr/bin/env python3
"""
Simple script to create PyTorch stub modules for deployment.
"""
import os
import site
import sys

def create_stub_modules():
    """Create necessary stub modules for PyTorch."""
    try:
        # Find the torch installation path
        import torch
        torch_path = os.path.dirname(torch.__file__)
        print(f"Found torch at: {torch_path}")
        
        # Create necessary directories
        dirs = [
            'cuda',
            'distributed',
            'distributed/rpc',
            'distributed/autograd'
        ]
        
        for d in dirs:
            full_path = os.path.join(torch_path, *d.split('/'))
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")
            
            # Create __init__.py in each directory
            init_file = os.path.join(full_path, '__init__.py')
            with open(init_file, 'w') as f:
                f.write(f"# Stub module for {d}\n")
                
                if d == 'cuda':
                    f.write("""
def is_available():
    return False

def device_count():
    return 0
""")
                elif d == 'distributed':
                    f.write("""
def is_initialized():
    return False
""")
                elif d == 'distributed/rpc':
                    f.write("""
def is_available():
    return False
""")
            
            print(f"Created stub module: {init_file}")
        
        print("Successfully created PyTorch stub modules")
        return True
    except Exception as e:
        print(f"Error creating stub modules: {e}")
        return False

if __name__ == '__main__':
    create_stub_modules()
EOF

# Modify the cleanup script to preserve our stubs
cat > preserve_stubs.py << 'EOF'
#!/usr/bin/env python3
"""
Script to modify cleanup.py to preserve stub modules.
"""
import os

def modify_cleanup_script():
    """Modify the cleanup script to preserve stub modules."""
    cleanup_path = 'cleanup.py'
    if not os.path.exists(cleanup_path):
        print(f"Warning: {cleanup_path} not found!")
        return False
    
    # Read the file
    with open(cleanup_path, 'r') as f:
        content = f.read()
    
    # Comment out lines that would remove our stub modules
    if "dirs_to_remove" in content:
        for module in ["'cuda'", '"cuda"', "'distributed'", '"distributed"']:
            if module in content:
                content = content.replace(
                    module,
                    f"# {module}  # Preserved for PyTorch initialization"
                )
    
    # Write back the modified content
    with open(cleanup_path, 'w') as f:
        f.write(content)
    
    print(f"Modified {cleanup_path} to preserve stub modules")
    return True

if __name__ == '__main__':
    modify_cleanup_script()
EOF

# Create PyTorch stub modules before cleanup
echo "Creating PyTorch stub modules..."
python create_torch_stubs.py

# Modify cleanup script to preserve stubs
echo "Modifying cleanup script..."
python preserve_stubs.py

# Fix application imports
echo "Fixing imports in app.py and chad.py..."
python fix_imports.py

# Now run the cleanup script (which has been modified to preserve stubs)
echo "Running cleanup script..."
python cleanup.py

# Verify torch imports still work
echo "Verifying PyTorch imports..."
python -c "import torch; print('PyTorch imports successful!')"

# Print final size of site-packages for debugging
du -sh $(python -c "import site; print(site.getsitepackages()[0])")

echo "Build completed successfully!"