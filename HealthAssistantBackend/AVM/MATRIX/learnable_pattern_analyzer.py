import torch
import torch.nn as nn
from typing import Dict, List, Any
from .cmdML import get_embedding, vectorQuotes

class LearnablePatternAnalyzer(nn.Module):
    """
    Learnable pattern analyzer that replaces hardcoded weights with learnable parameters.
    """
    
    def __init__(self, embedding_dim: int = 1536, vocab_size: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable word embeddings for severity/risk terms
        self.severity_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.risk_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Learnable severity classifier
        self.severity_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Severity score [0, 1]
            nn.Sigmoid()
        )
        
        # Learnable risk classifier
        self.risk_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Risk score [0, 1]
            nn.Sigmoid()
        )
        
        # Learnable vector store interpreter
        self.vector_interpreter = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),  # Query + Retrieved doc
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Context severity modifier
            nn.Tanh()  # [-1, 1] to modify base score
        )
        
        # Word-to-index mapping (learned during training)
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = vocab_size
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for idx, (word, freq) in enumerate(sorted_words[:self.vocab_size]):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def text_to_embeddings(self, text: str) -> torch.Tensor:
        """Convert text to learnable embeddings"""
        words = text.lower().split()
        word_indices = []
        
        for word in words:
            if word in self.word_to_idx:
                word_indices.append(self.word_to_idx[word])
            else:
                word_indices.append(0)  # Unknown word
        
        if not word_indices:
            return torch.zeros(self.embedding_dim)
        
        # Get embeddings for all words
        indices = torch.tensor(word_indices, dtype=torch.long)
        severity_emb = self.severity_embeddings(indices)
        risk_emb = self.risk_embeddings(indices)
        
        # Average embeddings
        avg_severity = torch.mean(severity_emb, dim=0)
        avg_risk = torch.mean(risk_emb, dim=0)
        
        return torch.cat([avg_severity, avg_risk])
    
    def analyze_patterns(self, text: str, vector_store_index=None) -> Dict[str, float]:
        """
        Analyze patterns with learnable parameters
        """
        # Get text embeddings
        text_emb = self.text_to_embeddings(text)
        
        # Get base severity and risk scores
        severity_score = self.severity_classifier(text_emb[:self.embedding_dim//2])
        risk_score = self.risk_classifier(text_emb[self.embedding_dim//2:])
        
        # Get vector store context if available
        if vector_store_index:
            try:
                # Get embedding for query
                query_embedding = get_embedding(text)
                relevant_docs = vectorQuotes(query_embedding, vector_store_index, top_k=3)
                
                # Process each relevant document
                context_modifier = 0.0
                for doc in relevant_docs:
                    doc_embedding = get_embedding(doc["text"])
                    
                    # Combine query and document embeddings
                    combined_emb = torch.cat([
                        torch.tensor(query_embedding, dtype=torch.float32),
                        torch.tensor(doc_embedding, dtype=torch.float32)
                    ])
                    
                    # Get context modifier
                    modifier = self.vector_interpreter(combined_emb)
                    context_modifier += modifier.item()
                
                # Average context modifier
                context_modifier /= len(relevant_docs)
                
                # Apply context modifier to scores
                severity_score = torch.clamp(severity_score + context_modifier * 0.2, 0, 1)
                risk_score = torch.clamp(risk_score + context_modifier * 0.2, 0, 1)
                
            except Exception as e:
                print(f"Vector store error: {e}")
        
        # Calculate agent confidences
        optimist_conf = 1.0 - ((severity_score + risk_score) / 2)
        pessimist_conf = (severity_score + risk_score) / 2
        
        return {
            "optimist_confidence": optimist_conf.item(),
            "pessimist_confidence": pessimist_conf.item(),
            "severity_score": severity_score.item(),
            "risk_score": risk_score.item()
        }
    
    def forward(self, text: str, vector_store_index=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        """
        text_emb = self.text_to_embeddings(text)
        
        severity_score = self.severity_classifier(text_emb[:self.embedding_dim//2])
        risk_score = self.risk_classifier(text_emb[self.embedding_dim//2:])
        
        # Get vector context if available
        context_modifier = torch.tensor(0.0)
        if vector_store_index:
            try:
                query_embedding = get_embedding(text)
                relevant_docs = vectorQuotes(query_embedding, vector_store_index, top_k=1)
                
                if relevant_docs:
                    doc_embedding = get_embedding(relevant_docs[0]["text"])
                    combined_emb = torch.cat([
                        torch.tensor(query_embedding, dtype=torch.float32),
                        torch.tensor(doc_embedding, dtype=torch.float32)
                    ])
                    context_modifier = self.vector_interpreter(combined_emb)
            except:
                pass
        
        return {
            "severity_score": severity_score,
            "risk_score": risk_score,
            "context_modifier": context_modifier,
            "text_embedding": text_emb
        }
