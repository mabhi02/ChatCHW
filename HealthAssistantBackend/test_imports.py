#!/usr/bin/env python3
"""
Test script to verify imports work without errors.
Run this locally with:
python test_imports.py
"""
import sys
print(f"Python version: {sys.version}")

print("Importing torch...")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available() if hasattr(torch, 'cuda') else False}")

print("Importing torch.nn...")
import torch.nn as nn
print("Successfully imported torch.nn")

print("Importing other modules required by the app...")
from groq import Groq
import pinecone
import openai
import flask
from flask_cors import CORS

print("All imports successful!")

# Try initializing MATRIX components
print("Trying to import app-specific modules...")
try:
    from chad import questions_init
    print("Successfully imported chad module!")
except Exception as e:
    print(f"Error importing chad module: {e}")
    import traceback
    traceback.print_exc()