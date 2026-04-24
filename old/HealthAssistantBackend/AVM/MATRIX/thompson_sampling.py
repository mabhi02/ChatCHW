import numpy as np
from typing import Dict

class AdaptiveThompsonSampler:
    def __init__(self):
        self.optimist_success = 1
        self.optimist_failure = 1
        self.pessimist_success = 1
        self.pessimist_failure = 1
        
    def sample_decision(self, 
                       optimist_view: Dict,
                       pessimist_view: Dict) -> str:
        """Thompson sampling for agent selection"""
        optimist_sample = np.random.beta(
            self.optimist_success,
            self.optimist_failure
        )
        pessimist_sample = np.random.beta(
            self.pessimist_success,
            self.pessimist_failure
        )
        
        return "optimist" if optimist_sample > pessimist_sample else "pessimist"
        
    def update(self, agent: str, success: bool):
        """Update beta distribution based on outcome"""
        if agent == "optimist":
            if success:
                self.optimist_success += 1
            else:
                self.optimist_failure += 1
        else:
            if success:
                self.pessimist_success += 1
            else:
                self.pessimist_failure += 1