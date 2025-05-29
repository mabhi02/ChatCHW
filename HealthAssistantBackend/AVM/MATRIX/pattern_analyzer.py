# pattern_analyzer.py

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import os
from pinecone import Pinecone
from .cmdML import get_embedding, vectorQuotes
from .config import MATRIXConfig

class PatternAnalyzer:
    """
    Analyzes medical text patterns to determine initial agent biases.
    Uses embeddings and vector DB lookups to assess severity and risk levels.
    """
    
    def __init__(self):
        self.embedding_dim = MATRIXConfig.EMBEDDING_DIM
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("who-guide-old")
        
        # Severity indicators with weights
        self.severity_indicators = {
            # High severity terms
            "severe": 0.8,
            "extreme": 0.9,
            "excruciating": 0.9,
            "unbearable": 0.9,
            "worst": 0.85,
            "intense": 0.8,
            # Moderate severity terms
            "moderate": 0.5,
            "significant": 0.6,
            "considerable": 0.6,
            "noticeable": 0.5,
            # Low severity terms
            "mild": 0.3,
            "slight": 0.2,
            "minor": 0.25,
            "light": 0.2,
            "manageable": 0.4
        }
        
        # Risk indicators with weights
        self.risk_indicators = {
            # High risk timing patterns
            "sudden": 0.8,
            "abrupt": 0.8,
            "immediate": 0.7,
            # Concerning durations
            "persistent": 0.6,
            "constant": 0.6,
            "continuous": 0.6,
            "worsening": 0.7,
            # Warning signs
            "unusual": 0.7,
            "concerning": 0.6,
            "abnormal": 0.7,
            "different": 0.5,
            "never": 0.6,
            "worried": 0.5,
            # Emergency indicators
            "emergency": 0.9,
            "critical": 0.9,
            "life-threatening": 1.0
        }

    def analyze_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze medical text for severity and risk patterns.
        Returns confidence scores for optimistic and pessimistic views.
        """
        try:
            # Get embedding and relevant medical knowledge
            embedding = get_embedding(text)
            relevant_docs = vectorQuotes(embedding, self.index, top_k=5)
            
            # Analyze patterns
            severity_score = self._analyze_severity(text, relevant_docs)
            risk_score = self._analyze_risk_factors(text, relevant_docs)
            
            # Calculate agent confidences
            optimist_conf = self._calculate_optimist_confidence(severity_score, risk_score)
            pessimist_conf = self._calculate_pessimist_confidence(severity_score, risk_score)
            
            return {
                "optimist_confidence": max(min(optimist_conf, 0.9), 0.1),
                "pessimist_confidence": max(min(pessimist_conf, 0.9), 0.1)
            }
            
        except Exception as e:
            print(f"Pattern analysis error: {str(e)}")
            # Return neutral confidences on error
            return {
                "optimist_confidence": 0.5,
                "pessimist_confidence": 0.5
            }
    
    def _analyze_severity(self, text: str, relevant_docs: List[Dict]) -> float:
        """Analyze text for severity indicators."""
        text = text.lower()
        
        # Start with neutral severity
        score = 0.5
        
        # Check direct severity indicators
        for term, weight in self.severity_indicators.items():
            if term in text:
                score = max(score, weight)
        
        # Analyze medical context if available
        if relevant_docs:
            for doc in relevant_docs:
                doc_text = doc["text"].lower()
                
                # Check for severity modifiers in medical context
                if any(term in doc_text for term in ["emergency", "immediate attention", "urgent"]):
                    score = min(score + 0.2, 1.0)
                if any(term in doc_text for term in ["common", "typical", "benign"]):
                    score = max(score - 0.1, 0.0)
                    
        return score
    
    def _analyze_risk_factors(self, text: str, relevant_docs: List[Dict]) -> float:
        """Analyze text for risk factors and warning signs."""
        text = text.lower()
        
        # Start with neutral risk
        score = 0.5
        
        # Check direct risk indicators
        for term, weight in self.risk_indicators.items():
            if term in text:
                score = max(score, weight)
        
        # Analyze medical context if available
        if relevant_docs:
            for doc in relevant_docs:
                doc_text = doc["text"].lower()
                
                # Check for risk modifiers in medical context
                if any(term in doc_text for term in ["warning sign", "red flag", "danger"]):
                    score = min(score + 0.2, 1.0)
                if any(term in doc_text for term in ["self-limiting", "harmless", "normal"]):
                    score = max(score - 0.1, 0.0)
                    
        return score
    
    def _calculate_optimist_confidence(self, severity: float, risk: float) -> float:
        """Calculate optimist confidence based on severity and risk scores."""
        # Optimist confidence decreases with severity and risk
        base_confidence = 1.0 - ((severity + risk) / 2)
        return base_confidence
    
    def _calculate_pessimist_confidence(self, severity: float, risk: float) -> float:
        """Calculate pessimist confidence based on severity and risk scores."""
        # Pessimist confidence increases with severity and risk
        base_confidence = (severity + risk) / 2
        return base_confidence

class AdaptiveAgent(nn.Module):
    """Base class for agents that adapt based on pattern analysis."""
    
    def __init__(self, is_optimist: bool):
        super().__init__()
        self.is_optimist = is_optimist
        self.pattern_analyzer = PatternAnalyzer()
        self.embedding_dim = MATRIXConfig.EMBEDDING_DIM
        
        # Initialize attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=MATRIXConfig.NUM_ATTENTION_HEADS,
            dropout=MATRIXConfig.DROPOUT_RATE,
            batch_first=True
        )
        
        # Initialize confidence dynamically
        self.confidence = 0.5
        
        # Initialize attention mask
        self.attention_mask = nn.Parameter(
            torch.ones((1, 1, self.embedding_dim))
        )
        
    def analyze_initial_state(self, initial_text: str):
        """Set initial confidence based on pattern analysis."""
        patterns = self.pattern_analyzer.analyze_patterns(initial_text)
        
        if self.is_optimist:
            self.confidence = patterns["optimist_confidence"]
            bias = 1.0 + self.confidence
        else:
            self.confidence = patterns["pessimist_confidence"]
            bias = 2.0 - self.confidence
            
        self.attention_mask.data = torch.ones((1, 1, self.embedding_dim)) * bias