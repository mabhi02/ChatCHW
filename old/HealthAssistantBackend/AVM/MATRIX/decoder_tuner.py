# decoder_tuner.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from .decoder import MATRIXDecoder

class DecoderTuner:
    """Uses ML to optimize decoder parameters based on outcomes"""
    
    def __init__(self, decoder: MATRIXDecoder):
        """Initialize tuner with decoder and optimization parameters"""
        self.decoder = decoder
        
        # Initialize optimizer with different learning rates for each component
        self.optimizer = optim.Adam([
            {'params': decoder.optimist_decoder.parameters(), 'lr': 1e-4},
            {'params': decoder.pessimist_decoder.parameters(), 'lr': 1e-4},
            {'params': decoder.optimist_projection.parameters(), 'lr': 1e-4},
            {'params': decoder.pessimist_projection.parameters(), 'lr': 1e-4},
        ])
        
        self.criterion = nn.MSELoss()
        self.history = []
        
    def prepare_batch(self, batch_X: torch.Tensor, batch_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for training by ensuring correct dimensions
        
        Args:
            batch_X: Input tensor of shape [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
            batch_y: Target tensor of shape [batch_size, 2] for optimist/pessimist weights
            
        Returns:
            Tuple of (prepared_X, prepared_y) with correct dimensions
        """
        if len(batch_X.shape) == 2:
            batch_X = batch_X.unsqueeze(1)  # Add sequence dimension [batch_size, 1, embedding_dim]
        return batch_X, batch_y
    
    def train_step(self, batch_X: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Perform a single training step
        
        Args:
            batch_X: Input tensor
            batch_y: Target tensor
            
        Returns:
            float: Loss value for this step
        """
        batch_X, batch_y = self.prepare_batch(batch_X, batch_y)
        
        self.optimizer.zero_grad()
        
        # Forward pass through decoder
        output = self.decoder(batch_X)
        
        # Calculate weights from outputs
        opt_weight = torch.mean(output['optimist_output'])
        pes_weight = torch.mean(output['pessimist_output'])
        pred_weights = torch.stack([opt_weight, pes_weight])
        
        # Calculate loss
        loss = self.criterion(pred_weights, batch_y.mean(dim=0))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, 
             X_train: torch.Tensor,
             y_train: torch.Tensor,
             X_val: Optional[torch.Tensor] = None,
             y_val: Optional[torch.Tensor] = None,
             epochs: int = 100,
             batch_size: int = 32) -> None:
        """
        Train decoder parameters
        
        Args:
            X_train: Training data tensor
            y_train: Training labels tensor
            X_val: Optional validation data tensor
            y_val: Optional validation labels tensor
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        use_validation = X_val is not None and y_val is not None
        
        for epoch in range(epochs):
            self.decoder.train()
            total_loss = 0
            n_batches = 0
            
            # Training loop
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                loss = self.train_step(batch_X, batch_y)
                total_loss += loss
                n_batches += 1
            
            # Validation if data is provided
            val_loss = None
            if use_validation:
                self.decoder.eval()
                with torch.no_grad():
                    try:
                        X_val_prep, y_val_prep = self.prepare_batch(X_val, y_val)
                        val_output = self.decoder(X_val_prep)
                        val_opt = torch.mean(val_output['optimist_output'])
                        val_pes = torch.mean(val_output['pessimist_output'])
                        val_pred = torch.stack([val_opt, val_pes])
                        val_loss = self.criterion(val_pred, y_val_prep.mean(dim=0)).item()
                    except Exception as e:
                        print(f"Warning: Validation failed: {str(e)}")
                        val_loss = None
            
            # Record history
            history_entry = {
                'epoch': epoch,
                'train_loss': total_loss / max(n_batches, 1),
            }
            if val_loss is not None:
                history_entry['val_loss'] = val_loss
            
            self.history.append(history_entry)
            
            # Log progress
            if epoch % 10 == 0:
                log_msg = f'Epoch {epoch}: Train Loss = {total_loss/max(n_batches, 1):.4f}'
                if val_loss is not None:
                    log_msg += f', Val Loss = {val_loss:.4f}'
                print(log_msg)
    
    def plot_training_history(self) -> plt.Figure:
        """
        Plot training and validation loss history
        
        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        plt.figure(figsize=(10, 5))
        
        epochs = [h['epoch'] for h in self.history]
        train_loss = [h['train_loss'] for h in self.history]
        
        plt.plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.7)
        
        # Plot validation loss if it exists
        if 'val_loss' in self.history[0]:
            val_loss = [h['val_loss'] for h in self.history]
            plt.plot(epochs, val_loss, label='Validation Loss', color='red', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Decoder Parameter Tuning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def save_model(self, path: str) -> None:
        """
        Save tuned decoder parameters
        
        Args:
            path: Path to save the model checkpoint
        """
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load tuned decoder parameters
        
        Args:
            path: Path to load the model checkpoint from
        """
        checkpoint = torch.load(path)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']