import torch
import torch.nn as nn
from typing import Dict, List, Any

class MATRIXDecoder(nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Dual attention decoders
        self.optimist_decoder = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        self.pessimist_decoder = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Projection layers
        self.optimist_projection = nn.Linear(embedding_dim, embedding_dim)
        self.pessimist_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, 
                state_embedding: torch.Tensor,
                memory: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Decode state embedding through dual attention paths
        """
        # If no memory provided, use state embedding
        if memory is None:
            memory = state_embedding
            
        # Get dual perspectives through separate decoders
        optimist_decoded = self.optimist_decoder(
            state_embedding,
            memory
        )
        
        pessimist_decoded = self.pessimist_decoder(
            state_embedding,
            memory
        )
        
        # Project to common space
        optimist_output = self.optimist_projection(optimist_decoded)
        pessimist_output = self.pessimist_projection(pessimist_decoded)
        
        return {
            "optimist_output": optimist_output,
            "pessimist_output": pessimist_output,
            "combined_output": (optimist_output + pessimist_output) / 2
        }