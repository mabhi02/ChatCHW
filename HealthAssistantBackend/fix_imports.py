import os

def fix_pinecone_imports():
    app_path = 'app.py'
    chad_path = 'chad.py'
    
    # Fix app.py
    if os.path.exists(app_path):
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Replace 'from pinecone import Pinecone' with 'import pinecone'
        modified_content = content.replace('from pinecone import Pinecone', 'import pinecone')
        # Replace 'pc = Pinecone(' with 'pc = pinecone.init('
        modified_content = modified_content.replace('pc = Pinecone(', 'pc = pinecone.init(')
        
        with open(app_path, 'w') as f:
            f.write(modified_content)
        print(f"Fixed imports in {app_path}")
    else:
        print(f"Could not find {app_path}")
    
    # Fix chad.py
    if os.path.exists(chad_path):
        with open(chad_path, 'r') as f:
            content = f.read()
        
        # Replace 'from pinecone import Pinecone' with 'import pinecone'
        modified_content = content.replace('from pinecone import Pinecone', 'import pinecone')
        # Replace 'pc = Pinecone(' with 'pc = pinecone.init('
        modified_content = modified_content.replace('pc = Pinecone(', 'pc = pinecone.init(')
        
        with open(chad_path, 'w') as f:
            f.write(modified_content)
        print(f"Fixed imports in {chad_path}")
    else:
        print(f"Could not find {chad_path}")

if __name__ == "__main__":
    print("Starting to fix imports...")
    fix_pinecone_imports()
    print("Done!")