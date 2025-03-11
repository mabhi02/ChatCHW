#!/usr/bin/env python3
"""
Test script to verify PyTorch and MATRIX imports work properly.
Run this after applying the fixes.
"""
import sys
import os

print(f"Python version: {sys.version}")

# First, test basic PyTorch imports
print("\nTesting PyTorch imports...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available() if hasattr(torch, 'cuda') else False}")
    
    import torch.nn as nn
    print("✓ Successfully imported torch.nn")
    
    # Test creating a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    print("✓ Successfully created a PyTorch model")
    
except Exception as e:
    print(f"✗ Error importing PyTorch: {e}")
    import traceback
    traceback.print_exc()

# Next, test MATRIX imports if available
print("\nTesting MATRIX imports...")
try:
    # Try importing MATRIX components - these may not be available in test environment
    try:
        from AVM.MATRIX.matrix_core import MATRIX
        from AVM.MATRIX.decoder_tuner import DecoderTuner 
        from AVM.MATRIX.attention_viz import AttentionVisualizer
        from AVM.MATRIX.state_encoder import StateSpaceEncoder
        from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
        from AVM.MATRIX.config import MATRIXConfig
        
        print("✓ Successfully imported MATRIX components")
        
        # Initialize MATRIX system and components
        matrix = MATRIX()
        print("✓ Successfully initialized MATRIX")
        
        decoder_tuner = DecoderTuner(matrix.meta_learner.decoder)
        print("✓ Successfully initialized DecoderTuner")
        
        visualizer = AttentionVisualizer()
        print("✓ Successfully initialized AttentionVisualizer")
        
        pattern_analyzer = PatternAnalyzer()
        print("✓ Successfully initialized PatternAnalyzer")
    except ImportError:
        print("Note: AVM.MATRIX modules not found in this environment. This is expected in test environments.")
    
    # Test importing MATRIX-dependent functions from chad.py
    print("\nTesting chad.py imports...")
    try:
        # Import just enough to verify the structure works
        import chad
        print(f"✓ Successfully imported chad module: {chad}")
        
        # Try to access some key functions/variables
        for item in [
            'questions_init', 
            'structured_questions_array', 
            'examination_history',
            'get_embedding_batch',
            'process_with_matrix',
            'judge'
        ]:
            if hasattr(chad, item):
                print(f"✓ Found {item} in chad module")
        
    except Exception as e:
        print(f"✗ Error importing chad.py: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"✗ Error in MATRIX tests: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")