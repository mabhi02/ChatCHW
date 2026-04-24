"""
MATRIX (Meta-learning Adaptive Thompson Reinforcement In X-space)
A dual-agent reasoning system.
"""

from .matrix_core import MATRIX, MetaLearningController
from .integration import MATRIXIntegration, MATRIXWrapper
from .agents import OptimistAgent, PessimistAgent
from .decoder import MATRIXDecoder
from .state_encoder import StateSpaceEncoder
from .thompson_sampling import AdaptiveThompsonSampler
from .config import MATRIXConfig
from .attention_viz import AttentionVisualizer
from .decoder_tuner import DecoderTuner

__all__ = [
    'MATRIX',
    'MATRIXIntegration',
    'MATRIXWrapper',
    'OptimistAgent',
    'PessimistAgent',
    'MATRIXDecoder',
    'StateSpaceEncoder',
    'AdaptiveThompsonSampler',
    'MetaLearningController',
    'MATRIXConfig',
    'AttentionVisualizer',
    'DecoderTuner'
]

__version__ = '0.1.0'