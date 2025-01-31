# attention_viz.py

import torch
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from .matrix_core import MATRIX

class AttentionVisualizer:
    """Visualizes attention patterns from MATRIX decoder"""
    
    def __init__(self):
        """Initialize with dark background style"""
        plt.style.use('dark_background')
    
    def plot_attention_weights(self, decoder_output: Dict[str, torch.Tensor], title: str):
        """Create heatmap of attention weights"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get attention weights and convert to numpy
        opt_attn = decoder_output["optimist_output"].detach().cpu().numpy()
        pes_attn = decoder_output["pessimist_output"].detach().cpu().numpy()
        
        # Reshape for visualization if needed
        if len(opt_attn.shape) == 3:  # [batch, seq, dim]
            opt_attn = opt_attn[0]  # Take first batch
        if len(pes_attn.shape) == 3:
            pes_attn = pes_attn[0]
            
        # Create simplified visualization (first 10 dimensions)
        opt_viz = opt_attn[:, :10]
        pes_viz = pes_attn[:, :10]
        
        # Create heatmaps
        sns.heatmap(opt_viz, ax=ax1, cmap='YlOrRd')
        sns.heatmap(pes_viz, ax=ax2, cmap='YlOrRd')
        
        ax1.set_title('Optimist Attention (First 10 dims)')
        ax2.set_title('Pessimist Attention (First 10 dims)')
        plt.suptitle(title)
        
        return fig

    def plot_comparative_focus(self, 
                             optimist_view: Dict[str, torch.Tensor],
                             pessimist_view: Dict[str, torch.Tensor],
                             text_tokens: List[str]):
        """Plot where each agent is focusing attention"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert attention states to numpy arrays and reduce dimensionality
        opt_state = optimist_view["state"].detach().cpu().numpy()
        pes_state = pessimist_view["state"].detach().cpu().numpy()
        
        # Calculate attention scores by taking mean across embedding dimension
        opt_focus = np.mean(opt_state, axis=2)[0]  # Shape becomes [seq_len]
        pes_focus = np.mean(pes_state, axis=2)[0]
        
        # Ensure we have the right number of tokens
        n_tokens = min(len(text_tokens), len(opt_focus))
        x = np.arange(n_tokens)
        width = 0.35
        
        # Create comparative bar plots
        ax.bar(x - width/2, opt_focus[:n_tokens], width, label='Optimist Focus', color='green', alpha=0.7)
        ax.bar(x + width/2, pes_focus[:n_tokens], width, label='Pessimist Focus', color='red', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(text_tokens[:n_tokens], rotation=45)
        ax.legend()
        
        plt.title('Agent Focus Comparison')
        plt.tight_layout()
        
        return fig

    def visualize_decoder_patterns(self,
                                 matrix: "MATRIX",
                                 text: str,
                                 save_path: Optional[str] = None):
        """Comprehensive visualization of decoder patterns"""
        # Tokenize text for visualization (simple space-based tokenization)
        tokens = text.split()
        
        # Create dummy responses for state encoding
        initial_responses = [{"question": "complaint", "answer": text}]
        followup_responses = []
        current_question = "followup"
        
        # Get state embedding
        state_embedding = matrix.state_encoder.encode_state(
            initial_responses,
            followup_responses,
            current_question
        )
        
        # Get decoder output
        decoder_output = matrix.meta_learner.decoder(state_embedding)
        
        # Create visualizations
        fig1 = self.plot_attention_weights(decoder_output, f'Attention Patterns for: "{text}"')
        
        # Get agent views
        opt_view = matrix.optimist.evaluate(state_embedding, text)
        pes_view = matrix.pessimist.evaluate(state_embedding, text)
        
        fig2 = self.plot_comparative_focus(opt_view, pes_view, tokens)
        
        if save_path:
            fig1.savefig(f'{save_path}_attention.png')
            fig2.savefig(f'{save_path}_focus.png')
        
        return fig1, fig2