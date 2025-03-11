import os
import re

def fix_pinecone_imports():
    app_path = os.path.join('HealthAssistantBackend', 'app.py')
    chad_path = os.path.join('HealthAssistantBackend', 'chad.py')
    
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

if __name__ == "__main__":
    fix_pinecone_imports()