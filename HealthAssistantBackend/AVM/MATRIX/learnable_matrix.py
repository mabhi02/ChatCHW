import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
from .learnable_pattern_analyzer import LearnablePatternAnalyzer
from .state_encoder import StateSpaceEncoder
from .thompson_sampling import AdaptiveThompsonSampler
from .config import MATRIXConfig

class LearnableMATRIX(nn.Module):
    """
    End-to-end learnable MATRIX system with proper backpropagation through all components.
    """
    
    def __init__(self, embedding_dim: int = 1536):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable components
        self.pattern_analyzer = LearnablePatternAnalyzer(embedding_dim)
        self.state_encoder = StateSpaceEncoder()
        
        # Learnable agents
        self.optimist_agent = LearnableAgent(is_optimist=True, embedding_dim=embedding_dim)
        self.pessimist_agent = LearnableAgent(is_optimist=False, embedding_dim=embedding_dim)
        
        # Learnable meta-controller
        self.meta_controller = LearnableMetaController(embedding_dim)
        
        # Learnable thresholds (replaces hardcoded values)
        self.similarity_threshold = nn.Parameter(torch.tensor(2.8))
        self.optimist_threshold = nn.Parameter(torch.tensor(0.7))
        self.max_questions = nn.Parameter(torch.tensor(5.0))
        
        # Thompson sampling (keeps RL component)
        self.thompson_sampler = AdaptiveThompsonSampler()
        
        # Optimizer for all learnable parameters
        self.optimizer = optim.Adam([
            {'params': self.pattern_analyzer.parameters(), 'lr': 1e-4},
            {'params': self.optimist_agent.parameters(), 'lr': 1e-4},
            {'params': self.pessimist_agent.parameters(), 'lr': 1e-4},
            {'params': self.meta_controller.parameters(), 'lr': 1e-4},
            {'params': [self.similarity_threshold, self.optimist_threshold, self.max_questions], 'lr': 1e-3}
        ])
        
        # Loss functions
        self.classification_loss = nn.BCELoss()
        self.regression_loss = nn.MSELoss()
        
    def forward(self, 
                initial_responses: List[Dict],
                followup_responses: List[Dict],
                current_question: str,
                vector_store_index=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire system
        """
        # 1. Pattern analysis
        initial_text = self._extract_initial_complaint(initial_responses)
        pattern_output = self.pattern_analyzer(initial_text, vector_store_index)
        
        # 2. State encoding
        state_embedding = self.state_encoder.encode_state(
            initial_responses, followup_responses, current_question
        )
        
        # 3. Agent evaluations
        optimist_output = self.optimist_agent(state_embedding, pattern_output)
        pessimist_output = self.pessimist_agent(state_embedding, pattern_output)
        
        # 4. Meta-controller combination
        combined_output = self.meta_controller(
            optimist_output, pessimist_output, state_embedding
        )
        
        # 5. Stopping decision
        should_stop = self._compute_stopping_decision(
            combined_output, len(followup_responses)
        )
        
        return {
            "optimist_output": optimist_output,
            "pessimist_output": pessimist_output,
            "combined_output": combined_output,
            "should_stop": should_stop,
            "pattern_output": pattern_output,
            "state_embedding": state_embedding
        }
    
    def _extract_initial_complaint(self, initial_responses: List[Dict]) -> str:
        """Extract initial complaint text"""
        for resp in initial_responses:
            if "complaint" in resp.get("question", "").lower():
                return resp.get("answer", "")
        return ""
    
    def _compute_stopping_decision(self, combined_output: Dict, num_questions: int) -> torch.Tensor:
        """Compute whether to stop asking questions"""
        confidence = combined_output["confidence"]
        optimist_weight = combined_output["weights"]["optimist"]
        
        # Learnable stopping criteria
        stop_by_confidence = confidence > torch.sigmoid(self.similarity_threshold)
        stop_by_optimist = optimist_weight > torch.sigmoid(self.optimist_threshold)
        stop_by_count = num_questions >= torch.sigmoid(self.max_questions) * 10  # Scale to reasonable range
        
        return torch.logical_or(
            torch.logical_or(stop_by_confidence, stop_by_optimist),
            stop_by_count
        )
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    ground_truth: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute end-to-end loss for training
        """
        losses = {}
        
        # 1. Classification loss (severe vs non-severe)
        predicted_severity = predictions["combined_output"]["severity_prob"]
        true_severity = torch.tensor(1.0 if ground_truth["is_severe"] else 0.0)
        losses["classification"] = self.classification_loss(predicted_severity, true_severity)
        
        # 2. Stopping decision loss
        predicted_stop = predictions["should_stop"]
        true_stop = torch.tensor(1.0 if ground_truth["should_have_stopped"] else 0.0)
        losses["stopping"] = self.classification_loss(predicted_stop, true_stop)
        
        # 3. Agent consistency loss
        optimist_conf = predictions["optimist_output"]["confidence"]
        pessimist_conf = predictions["pessimist_output"]["confidence"]
        losses["consistency"] = self.regression_loss(optimist_conf + pessimist_conf, torch.tensor(1.0))
        
        # 4. Pattern analysis loss
        predicted_severity_score = predictions["pattern_output"]["severity_score"]
        true_severity_score = torch.tensor(ground_truth["severity_score"])
        losses["pattern"] = self.regression_loss(predicted_severity_score, true_severity_score)
        
        # 5. Efficiency loss (encourage early stopping when possible)
        if ground_truth["is_severe"] == ground_truth["should_have_stopped"]:
            # If we made the right decision, encourage efficiency
            losses["efficiency"] = torch.mean(predictions["combined_output"]["confidence"])
        else:
            # If we made the wrong decision, don't penalize efficiency
            losses["efficiency"] = torch.tensor(0.0)
        
        return losses
    
    def train_step(self, 
                  batch_data: List[Dict],
                  vector_store_index=None) -> Dict[str, float]:
        """
        Single training step
        """
        self.optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0)
        batch_losses = []
        
        for case in batch_data:
            # Forward pass
            predictions = self.forward(
                case["initial_responses"],
                case["followup_responses"],
                case["current_question"],
                vector_store_index
            )
            
            # Compute loss
            losses = self.compute_loss(predictions, case["ground_truth"])
            case_loss = sum(losses.values())
            total_loss += case_loss
            batch_losses.append(losses)
        
        # Average loss
        avg_loss = total_loss / len(batch_data)
        
        # Backward pass
        avg_loss.backward()
        self.optimizer.step()
        
        # Update Thompson sampling based on outcomes
        for case, losses in zip(batch_data, batch_losses):
            success = case["ground_truth"]["is_severe"] == case["ground_truth"]["should_have_stopped"]
            selected_agent = "optimist" if predictions["combined_output"]["weights"]["optimist"] > 0.5 else "pessimist"
            self.thompson_sampler.update(selected_agent, success)
        
        return {
            "total_loss": avg_loss.item(),
            "classification_loss": torch.mean(torch.tensor([l["classification"].item() for l in batch_losses])),
            "stopping_loss": torch.mean(torch.tensor([l["stopping"].item() for l in batch_losses])),
            "pattern_loss": torch.mean(torch.tensor([l["pattern"].item() for l in batch_losses]))
        }
    
    def predict(self, 
               initial_responses: List[Dict],
               followup_responses: List[Dict],
               current_question: str,
               vector_store_index=None) -> Dict[str, Any]:
        """
        Make prediction (inference mode)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(
                initial_responses, followup_responses, current_question, vector_store_index
            )
        
        return {
            "is_severe": predictions["combined_output"]["severity_prob"].item() > 0.5,
            "confidence": predictions["combined_output"]["confidence"].item(),
            "should_stop": predictions["should_stop"].item() > 0.5,
            "optimist_weight": predictions["combined_output"]["weights"]["optimist"].item(),
            "pessimist_weight": predictions["combined_output"]["weights"]["pessimist"].item(),
            "selected_agent": "optimist" if predictions["combined_output"]["weights"]["optimist"] > 0.5 else "pessimist"
        }

class LearnableAgent(nn.Module):
    """Learnable agent with attention mechanisms"""
    
    def __init__(self, is_optimist: bool, embedding_dim: int = 1536):
        super().__init__()
        self.is_optimist = is_optimist
        self.embedding_dim = embedding_dim
        
        # Learnable attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable projection
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
        
        # Learnable confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Learnable attention bias (replaces hardcoded mask)
        self.attention_bias = nn.Parameter(torch.ones(embedding_dim))
        
    def forward(self, state_embedding: torch.Tensor, pattern_output: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass through agent"""
        # Apply learnable attention bias
        biased_state = state_embedding * self.attention_bias.unsqueeze(0).unsqueeze(0)
        
        # Multi-head attention
        attended_state, _ = self.attention(
            biased_state, biased_state, biased_state
        )
        
        # Project and activate
        projected_state = self.projection(attended_state)
        activated_state = self.activation(projected_state)
        
        # Predict confidence
        confidence = self.confidence_predictor(activated_state.mean(dim=1))
        
        return {
            "state": activated_state,
            "confidence": confidence,
            "attention_weights": self.attention_bias
        }

class LearnableMetaController(nn.Module):
    """Learnable meta-controller for combining agent outputs"""
    
    def __init__(self, embedding_dim: int = 1536):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable combination weights
        self.combination_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # optimist + pessimist states
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # optimist_weight, pessimist_weight
        )
        
        # Learnable confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Learnable severity predictor
        self.severity_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                optimist_output: Dict[str, torch.Tensor],
                pessimist_output: Dict[str, torch.Tensor],
                state_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Combine agent outputs"""
        # Concatenate agent states
        combined_states = torch.cat([
            optimist_output["state"].mean(dim=1),
            pessimist_output["state"].mean(dim=1)
        ], dim=1)
        
        # Predict combination weights
        raw_weights = self.combination_network(combined_states)
        weights = torch.softmax(raw_weights, dim=1)
        
        # Combine states
        combined_state = (
            optimist_output["state"] * weights[:, 0:1].unsqueeze(-1) +
            pessimist_output["state"] * weights[:, 1:2].unsqueeze(-1)
        )
        
        # Predict final outputs
        confidence = self.confidence_predictor(combined_state.mean(dim=1))
        severity_prob = self.severity_predictor(combined_state.mean(dim=1))
        
        return {
            "state": combined_state,
            "confidence": confidence,
            "severity_prob": severity_prob,
            "weights": {
                "optimist": weights[:, 0],
                "pessimist": weights[:, 1]
            }
        }
