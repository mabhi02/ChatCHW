# config.py

class MATRIXConfig:
    """
    Configuration parameters for the MATRIX (Meta-learning Adaptive Thompson 
    Reinforcement In X-space) system.
    """
    
    # Model Architecture Parameters
    EMBEDDING_DIM = 1536  # Updated to match OpenAI's text-embedding-3-small dimension
    NUM_ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.1
    
    # Agent Behavior Parameters
    OPTIMIST_BASE_CONFIDENCE = 0.6
    PESSIMIST_BASE_CONFIDENCE = 0.4
    OPTIMIST_ATTENTION_BIAS = 1.2  # Amplifies positive features
    PESSIMIST_ATTENTION_BIAS = 0.8  # Amplifies warning signs
    
    # Thompson Sampling Parameters
    INITIAL_BETA_PARAMS = 1.0  # Starting value for success/failure counts
    THOMPSON_TEMPERATURE = 0.1  # Exploration-exploitation trade-off
    
    # Similarity Thresholds (matched to cmdML.py)
    SIMILARITY_THRESHOLD = 2.8  # For question similarity judgment
    EXAM_SIMILARITY_THRESHOLD = 1.5  # For examination similarity judgment
    MAX_EXAMS = 5  # Maximum number of examinations
    
    # Learning Parameters
    META_LEARNING_RATE = 0.01  # For meta-learning updates
    ATTENTION_LEARNING_RATE = 0.001  # For attention mechanism updates
    CONFIDENCE_UPDATE_RATE = 0.1  # For agent confidence updates
    
    # System Constraints
    MIN_CONFIDENCE = 0.1  # Minimum agent confidence
    MAX_CONFIDENCE = 0.9  # Maximum agent confidence
    MAX_QUESTIONS = 10  # Maximum number of questions
    MAX_FOLLOWUPS = 5  # Maximum number of follow-up questions
    
    # State Space Parameters
    STATE_UNCERTAINTY_THRESHOLD = 0.3  # Threshold for state uncertainty
    CONFIDENCE_SCALING = 1.2  # Scaling factor for confidence updates
    
    # Memory Parameters
    MEMORY_SIZE = 1000  # Size of experience memory
    BATCH_SIZE = 32  # Batch size for updates
    
    # Adaptation Parameters
    ADAPTATION_THRESHOLD = 0.1  # Threshold for adapting behavior
    MAX_ADAPTATION_STEPS = 100  # Maximum adaptation iterations
    
    @classmethod
    def get_agent_params(cls) -> dict:
        """Get all agent-related parameters"""
        return {
            "embedding_dim": cls.EMBEDDING_DIM,
            "num_heads": cls.NUM_ATTENTION_HEADS,
            "dropout": cls.DROPOUT_RATE,
            "optimist_bias": cls.OPTIMIST_ATTENTION_BIAS,
            "pessimist_bias": cls.PESSIMIST_ATTENTION_BIAS
        }
    
    @classmethod
    def get_learning_params(cls) -> dict:
        """Get all learning-related parameters"""
        return {
            "meta_lr": cls.META_LEARNING_RATE,
            "attention_lr": cls.ATTENTION_LEARNING_RATE,
            "confidence_update": cls.CONFIDENCE_UPDATE_RATE,
            "min_confidence": cls.MIN_CONFIDENCE,
            "max_confidence": cls.MAX_CONFIDENCE
        }
    
    @classmethod
    def get_threshold_params(cls) -> dict:
        """Get all threshold-related parameters"""
        return {
            "similarity": cls.SIMILARITY_THRESHOLD,
            "exam_similarity": cls.EXAM_SIMILARITY_THRESHOLD,
            "max_exams": cls.MAX_EXAMS,
            "uncertainty": cls.STATE_UNCERTAINTY_THRESHOLD,
            "adaptation": cls.ADAPTATION_THRESHOLD
        }