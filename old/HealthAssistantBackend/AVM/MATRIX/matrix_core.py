# matrix_core.py

import torch
from typing import Dict, List, Any
from .state_encoder import StateSpaceEncoder
from .agents import OptimistAgent, PessimistAgent
from .thompson_sampling import AdaptiveThompsonSampler
from .decoder import MATRIXDecoder
from .config import MATRIXConfig

class MetaLearningController:
    """Controls the meta-learning process and combines agent views"""
    def __init__(self, embedding_dim: int = MATRIXConfig.EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.decoder = MATRIXDecoder(embedding_dim)
        self.meta_lr = MATRIXConfig.META_LEARNING_RATE
        
    def process(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process state through dual attention paths"""
        return self.decoder(state)
        
    def combine_views(self,
                     optimist_view: Dict,
                     pessimist_view: Dict,
                     selected_view: str) -> Dict:
        """Combine agent views based on selection and confidence"""
        # Get states and confidences
        opt_state = optimist_view["state"]
        pes_state = pessimist_view["state"]
        opt_conf = optimist_view["confidence"]
        pes_conf = pessimist_view["confidence"]
        
        # Dynamic weighting based on confidences
        total_conf = opt_conf + pes_conf
        if total_conf > 0:
            opt_weight = opt_conf / total_conf
            pes_weight = pes_conf / total_conf
        else:
            opt_weight = pes_weight = 0.5
            
        # Additional bias based on Thompson sampling selection
        if selected_view == "optimist":
            opt_weight *= 1.2
            pes_weight *= 0.8
        else:
            opt_weight *= 0.8
            pes_weight *= 1.2
            
        # Normalize weights
        weight_sum = opt_weight + pes_weight
        opt_weight /= weight_sum
        pes_weight /= weight_sum
        
        # Combine states with weights
        combined_state = (
            opt_state * opt_weight +
            pes_state * pes_weight
        )
        
        # Calculate final confidence
        combined_confidence = max(opt_conf, pes_conf)
        
        return {
            "state": combined_state,
            "confidence": combined_confidence,
            "selected_agent": selected_view,
            "weights": {
                "optimist": opt_weight,
                "pessimist": pes_weight
            }
        }

class MATRIX:
    """
    MATRIX (Meta-learning Adaptive Thompson Reinforcement In X-space) system.
    Combines optimistic and pessimistic views for medical diagnosis.
    """
    def __init__(self):
        self.state_encoder = StateSpaceEncoder()
        self.meta_learner = MetaLearningController()
        self.thompson_sampler = AdaptiveThompsonSampler()
        self.optimist = OptimistAgent()
        self.pessimist = PessimistAgent()
        
        # Confidence thresholds
        self.similarity_threshold = MATRIXConfig.SIMILARITY_THRESHOLD
        self.exam_similarity_threshold = MATRIXConfig.EXAM_SIMILARITY_THRESHOLD
        
    def process_state(self, 
                     initial_responses: List[Dict],
                     followup_responses: List[Dict],
                     current_question: str) -> Dict:
        """
        Core MATRIX processing pipeline with dynamic agent adaptation
        """
        # Get initial complaint text for pattern analysis
        initial_text = next((resp['answer'] for resp in initial_responses 
                           if resp['question'] == "Please describe what brings you here today"), "")
        
        # Encode current state
        state_embedding = self.state_encoder.encode_state(
            initial_responses, 
            followup_responses,
            current_question
        )
        
        # Get dual perspectives with initial text analysis
        optimist_view = self.optimist.evaluate(state_embedding, initial_text)
        pessimist_view = self.pessimist.evaluate(state_embedding, initial_text)
        
        # Use Thompson sampling for decision
        selected_view = self.thompson_sampler.sample_decision(
            optimist_view,
            pessimist_view
        )
        
        # Combine views and get final decision
        result = self.meta_learner.combine_views(
            optimist_view,
            pessimist_view,
            selected_view
        )
        
        # Store current views for learning
        self.last_optimist_view = optimist_view
        self.last_pessimist_view = pessimist_view
        
        return result
        
    def update_from_feedback(self, success: bool):
        """Update agents based on feedback"""
        # Update agent that made the decision
        if hasattr(self, 'last_optimist_view') and hasattr(self, 'last_pessimist_view'):
            if self.last_optimist_view["confidence"] > self.last_pessimist_view["confidence"]:
                self.optimist.update_confidence(success)
            else:
                self.pessimist.update_confidence(success)
            
        # Update Thompson sampling
        self.thompson_sampler.update(
            "optimist" if self.last_optimist_view["confidence"] > self.last_pessimist_view["confidence"] else "pessimist",
            success
        )