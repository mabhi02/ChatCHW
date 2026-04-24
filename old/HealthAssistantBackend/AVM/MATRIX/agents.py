# agents.py

import torch
import torch.nn as nn
from typing import Dict, List, Any
from .config import MATRIXConfig
from .pattern_analyzer import PatternAnalyzer, AdaptiveAgent

class OptimistAgent(AdaptiveAgent):
    """
    Optimist agent that tends to focus on positive indicators in the state space.
    Uses attention mechanisms with positive bias to evaluate patient state.
    """
    def __init__(self):
        super().__init__(is_optimist=True)
        self.name = "optimist"
        self.embedding_dim = MATRIXConfig.EMBEDDING_DIM
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=MATRIXConfig.NUM_ATTENTION_HEADS,
            dropout=MATRIXConfig.DROPOUT_RATE,
            batch_first=True
        )
        
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.activation = nn.ReLU()
        
        # Initialize with neutral attention mask
        self.attention_mask = nn.Parameter(
            torch.ones((1, 1, self.embedding_dim))
        )
        
        self.confidence = 0.5  # Start neutral until analysis
        
    def analyze_initial_state(self, initial_text: str):
        """Analyze initial state to set confidence and attention bias"""
        patterns = self.pattern_analyzer.analyze_patterns(initial_text)
        self.confidence = patterns["optimist_confidence"]
        
        # Update attention mask based on analysis
        bias = 1.0 + self.confidence
        self.attention_mask.data = torch.ones((1, 1, self.embedding_dim)) * bias

    def evaluate(self, state_embedding: torch.Tensor, initial_text: str = None) -> Dict[str, Any]:
        """Evaluate state with dynamic optimistic bias"""
        if initial_text:
            self.analyze_initial_state(initial_text)
            
        # Ensure input shape is (batch_size, seq_len, embedding_dim)
        if len(state_embedding.shape) == 2:
            state_embedding = state_embedding.unsqueeze(1)
        elif len(state_embedding.shape) == 1:
            state_embedding = state_embedding.view(1, 1, -1)
            
        # Apply attention mask
        masked_state = state_embedding * self.attention_mask.to(state_embedding.device)
            
        # Multi-head attention
        attended_state, _ = self.attention(
            masked_state,
            masked_state,
            masked_state
        )
        
        # Project and activate
        projected_state = self.projection(attended_state)
        activated_state = self.activation(projected_state)
        
        return {
            "confidence": self.confidence,
            "state": activated_state,
            "name": self.name
        }
        
    def update_confidence(self, success: bool):
        """Update confidence based on feedback"""
        if success:
            self.confidence = min(
                MATRIXConfig.MAX_CONFIDENCE,
                self.confidence + MATRIXConfig.CONFIDENCE_UPDATE_RATE
            )
        else:
            self.confidence = max(
                MATRIXConfig.MIN_CONFIDENCE,
                self.confidence - MATRIXConfig.CONFIDENCE_UPDATE_RATE
            )
            
        # Update attention mask with new confidence
        bias = 1.0 + self.confidence
        self.attention_mask.data = torch.ones((1, 1, self.embedding_dim)) * bias

class PessimistAgent(AdaptiveAgent):
    """
    Pessimist agent that tends to focus on warning signs and potential risks.
    Uses attention mechanisms with negative bias to evaluate patient state.
    """
    def __init__(self):
        super().__init__(is_optimist=False)
        self.name = "pessimist"
        self.embedding_dim = MATRIXConfig.EMBEDDING_DIM
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=MATRIXConfig.NUM_ATTENTION_HEADS,
            dropout=MATRIXConfig.DROPOUT_RATE,
            batch_first=True
        )
        
        self.projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.activation = nn.ReLU()
        
        # Initialize with neutral attention mask
        self.attention_mask = nn.Parameter(
            torch.ones((1, 1, self.embedding_dim))
        )
        
        self.confidence = 0.5  # Start neutral until analysis
        
    def analyze_initial_state(self, initial_text: str):
        """Analyze initial state to set confidence and attention bias"""
        patterns = self.pattern_analyzer.analyze_patterns(initial_text)
        self.confidence = patterns["pessimist_confidence"]
        
        # Update attention mask based on analysis
        bias = 2.0 - self.confidence  # Inverse relationship for pessimist
        self.attention_mask.data = torch.ones((1, 1, self.embedding_dim)) * bias

    def evaluate(self, state_embedding: torch.Tensor, initial_text: str = None) -> Dict[str, Any]:
        """Evaluate state with dynamic pessimistic bias"""
        if initial_text:
            self.analyze_initial_state(initial_text)
            
        # Ensure input shape is (batch_size, seq_len, embedding_dim)
        if len(state_embedding.shape) == 2:
            state_embedding = state_embedding.unsqueeze(1)
        elif len(state_embedding.shape) == 1:
            state_embedding = state_embedding.view(1, 1, -1)
            
        # Apply attention mask
        masked_state = state_embedding * self.attention_mask.to(state_embedding.device)
            
        # Multi-head attention
        attended_state, _ = self.attention(
            masked_state,
            masked_state,
            masked_state
        )
        
        # Project and activate
        projected_state = self.projection(attended_state)
        activated_state = self.activation(projected_state)
        
        return {
            "confidence": self.confidence,
            "state": activated_state,
            "name": self.name
        }
        
    def update_confidence(self, success: bool):
        """Update confidence based on feedback"""
        if success:
            self.confidence = min(
                MATRIXConfig.MAX_CONFIDENCE,
                self.confidence + MATRIXConfig.CONFIDENCE_UPDATE_RATE
            )
        else:
            self.confidence = max(
                MATRIXConfig.MIN_CONFIDENCE,
                self.confidence - MATRIXConfig.CONFIDENCE_UPDATE_RATE
            )
            
        # Update attention mask with new confidence
        bias = 2.0 - self.confidence  # Inverse relationship for pessimist
        self.attention_mask.data = torch.ones((1, 1, self.embedding_dim)) * bias