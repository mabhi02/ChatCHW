import os
import shutil
import site
import sys

def cleanup_torch():
    """Remove unnecessary torch components to reduce size"""
    try:
        site_packages = site.getsitepackages()[0]
        torch_path = os.path.join(site_packages, 'torch')
        
        # Directories to remove from torch
        dirs_to_remove = [
            'test',
            'testing',
            'optim/test',
            'nn/test',
            'distributions/test',
            'cuda',  # Remove CUDA support as we're using CPU only
            'utils/cpp_extension.py',  # Remove C++ extension utilities
            'distributed',  # Remove distributed training support if not needed
        ]
        
        for dir_name in dirs_to_remove:
            dir_path = os.path.join(torch_path, dir_name)
            if os.path.exists(dir_path):
                print(f"Removing {dir_path}")
                try:
                    if os.path.isdir(dir_path):
                        shutil.rmtree(dir_path)
                    else:
                        os.remove(dir_path)
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
                    
        # Remove .html files in torch docs
        for root, dirs, files in os.walk(torch_path):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
    except Exception as e:
        print(f"Error in cleanup_torch: {e}")

def cleanup_matplotlib():
    """Remove matplotlib test data and examples"""
    try:
        site_packages = site.getsitepackages()[0]
        mpl_path = os.path.join(site_packages, 'matplotlib')
        
        dirs_to_remove = [
            'tests',
            'testing',
            'examples',
        ]
        
        for dir_name in dirs_to_remove:
            dir_path = os.path.join(mpl_path, dir_name)
            if os.path.exists(dir_path):
                print(f"Removing {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
        
        # Remove sample data
        sample_data = os.path.join(mpl_path, 'mpl-data', 'sample_data')
        if os.path.exists(sample_data):
            print(f"Removing {sample_data}")
            try:
                shutil.rmtree(sample_data)
            except Exception as e:
                print(f"Error removing {sample_data}: {e}")
    except Exception as e:
        print(f"Error in cleanup_matplotlib: {e}")

def cleanup_scikit():
    """Remove scikit-learn datasets and test files"""
    try:
        site_packages = site.getsitepackages()[0]
        sklearn_path = os.path.join(site_packages, 'sklearn')
        
        dirs_to_remove = [
            'datasets',  # Remove bundled datasets
            '__pycache__',
            'tests',
        ]
        
        for dir_name in dirs_to_remove:
            dir_path = os.path.join(sklearn_path, dir_name)
            if os.path.exists(dir_path):
                print(f"Removing {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error removing {dir_path}: {e}")
    except Exception as e:
        print(f"Error in cleanup_scikit: {e}")

def cleanup_pyc_files():
    """Remove all .pyc files to reduce size"""
    try:
        site_packages = site.getsitepackages()[0]
        count = 0
        
        for root, dirs, files in os.walk(site_packages):
            for file in files:
                if file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        count += 1
                    except:
                        pass
        
        print(f"Removed {count} .pyc files")
    except Exception as e:
        print(f"Error in cleanup_pyc_files: {e}")

if __name__ == '__main__':
    print("Starting cleanup...")
    cleanup_torch()
    cleanup_matplotlib()
    cleanup_scikit()
    cleanup_pyc_files()
    print("Cleanup completed!")