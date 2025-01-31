import torch
from typing import Dict, List
import numpy as np
from .cmdML import get_embedding
from .config import MATRIXConfig

class StateSpaceEncoder:
    def __init__(self):
        self.embedding_dim = MATRIXConfig.EMBEDDING_DIM
        
    def encode_state(self, 
                    initial_responses: List[Dict],
                    followup_responses: List[Dict],
                    current_question: str) -> torch.Tensor:
        """
        Encode the current state into state space
        """
        # Format responses into text
        initial_text = self._format_responses(initial_responses)
        followup_text = self._format_responses(followup_responses)
        
        # Get embeddings using existing get_embedding function
        initial_embedding = get_embedding(initial_text)
        followup_embedding = get_embedding(followup_text)
        current_embedding = get_embedding(current_question)
        
        # Ensure all embeddings have correct dimension
        if len(initial_embedding) != self.embedding_dim:
            print(f"Warning: Initial embedding dimension {len(initial_embedding)} does not match expected {self.embedding_dim}")
            # Pad or truncate to match expected dimension
            initial_embedding = initial_embedding[:self.embedding_dim]
        
        if len(followup_embedding) != self.embedding_dim:
            print(f"Warning: Followup embedding dimension {len(followup_embedding)} does not match expected {self.embedding_dim}")
            followup_embedding = followup_embedding[:self.embedding_dim]
            
        if len(current_embedding) != self.embedding_dim:
            print(f"Warning: Current embedding dimension {len(current_embedding)} does not match expected {self.embedding_dim}")
            current_embedding = current_embedding[:self.embedding_dim]
        
        # Combine embeddings
        combined_state = self._combine_embeddings(
            initial_embedding,
            followup_embedding,
            current_embedding
        )
        
        return combined_state
        
    def _format_responses(self, responses: List[Dict]) -> str:
        """Format responses into text for embedding"""
        formatted = []
        for resp in responses:
            formatted.append(f"Q: {resp['question']}\nA: {resp['answer']}")
        return "\n".join(formatted)
        
    def _combine_embeddings(self,
                          initial_embedding: List[float],
                          followup_embedding: List[float],
                          current_embedding: List[float]) -> torch.Tensor:
        """Combine different embeddings into state representation"""
        # Convert to tensors and ensure correct shape
        initial = torch.tensor(initial_embedding, dtype=torch.float32)[:self.embedding_dim]
        followup = torch.tensor(followup_embedding, dtype=torch.float32)[:self.embedding_dim]
        current = torch.tensor(current_embedding, dtype=torch.float32)[:self.embedding_dim]
        
        # Weight the combinations
        state = (
            initial * 0.3 +  # Initial responses get 30% weight
            followup * 0.4 + # Followup gets 40% weight
            current * 0.3    # Current question gets 30% weight
        )
        
        # Normalize
        state = state / torch.norm(state)
        
        # Reshape to (batch_size, seq_len, embedding_dim) for attention
        state = state.view(1, 1, self.embedding_dim)
        
        return state