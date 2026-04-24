import os
from dotenv import load_dotenv
import torch
from MATRIX.matrix_core import MATRIX
from MATRIX.agents import OptimistAgent, PessimistAgent
from MATRIX.config import MATRIXConfig
from MATRIX.pattern_analyzer import PatternAnalyzer

from MATRIX.attention_viz import AttentionVisualizer
from MATRIX.decoder_tuner import DecoderTuner

def test_pattern_analysis():
    """Test pattern analyzer with different medical scenarios"""
    print("\nTesting Pattern Analysis...")
    analyzer = PatternAnalyzer()
    
    test_cases = [
        {
            "description": "Severe Case",
            "text": "Severe headache with sudden onset, worst pain ever experienced, worried about stroke"
        },
        {
            "description": "Moderate Case",
            "text": "Moderate fever and cough for two days, manageable with over-the-counter medication"
        },
        {
            "description": "Mild Case",
            "text": "Mild joint pain in knees, slightly worse after exercise but generally manageable"
        },
        {
            "description": "Emergency Case",
            "text": "Sudden chest pain radiating to left arm, difficulty breathing, emergency situation"
        }
    ]
    
    for case in test_cases:
        print(f"\nAnalyzing: {case['description']}")
        print(f"Text: '{case['text']}'")
        
        patterns = analyzer.analyze_patterns(case['text'])
        
        print("Analysis Results:")
        print(f"Optimist Confidence: {patterns['optimist_confidence']:.3f}")
        print(f"Pessimist Confidence: {patterns['pessimist_confidence']:.3f}")
        
def test_agents():
    """Test the optimist and pessimist agents with dynamic pattern analysis"""
    print("\nTesting Agents with Pattern Analysis...")
    
    # Create test state embedding
    test_embedding = torch.randn(1, MATRIXConfig.EMBEDDING_DIM)
    test_text = "Severe headache, sudden onset, with nausea and sensitivity to light"
    
    # Test Optimist Agent
    print("\nOptimist Agent Analysis:")
    optimist = OptimistAgent()
    optimist_result = optimist.evaluate(test_embedding, test_text)
    print(f"Initial Confidence: {optimist.confidence:.3f}")
    print(f"After Analysis Confidence: {optimist_result['confidence']:.3f}")
    
    # Test Pessimist Agent
    print("\nPessimist Agent Analysis:")
    pessimist = PessimistAgent()
    pessimist_result = pessimist.evaluate(test_embedding, test_text)
    print(f"Initial Confidence: {pessimist.confidence:.3f}")
    print(f"After Analysis Confidence: {pessimist_result['confidence']:.3f}")

def test_matrix_pipeline():
    """Test the full MATRIX pipeline with visualization and tuning"""
    print("\nTesting Enhanced MATRIX Pipeline...")
    
    matrix = MATRIX()
    visualizer = AttentionVisualizer()
    tuner = DecoderTuner(matrix.meta_learner.decoder)
    
    # Test cases for training
    training_cases = [
        {
            "text": "Severe chest pain with shortness of breath",
            "outcome": "success",  # Pessimist was right to be concerned
            "expected_agent": "pessimist"
        },
        {
            "text": "Mild headache, responds well to OTC medication",
            "outcome": "success",  # Optimist was right about benign nature
            "expected_agent": "optimist"
        }
    ]
    
    # Process cases and collect outcomes
    case_outcomes = []
    
    for case in training_cases:
        # Create initial responses structure for state encoding
        initial_responses = [
            {
                "question": "Please describe what brings you here today",
                "answer": case["text"],
                "type": "FREE"
            }
        ]
        
        # Get state embedding using the correct method
        state_embedding = matrix.state_encoder.encode_state(
            initial_responses,
            [],  # Empty followup responses
            "followup"  # Default current question
        )
        
        # Process through MATRIX
        result = matrix.process_state(
            initial_responses,
            [],
            "followup"
        )
        
        # Store outcome
        case_outcomes.append({
            "state_embedding": state_embedding,
            "outcome": case["outcome"],
            "selected_agent": result["selected_agent"],
            "expected_agent": case["expected_agent"]
        })
        
        # Visualize attention patterns
        print(f"\nProcessing case: {case['text']}")
        visualizer.visualize_decoder_patterns(matrix, case["text"])
    
    # Prepare training data
    X_train = torch.stack([outcome["state_embedding"].squeeze() for outcome in case_outcomes])
    y_train = torch.tensor([[1, 0] if outcome["expected_agent"] == "optimist" else [0, 1] 
                         for outcome in case_outcomes], dtype=torch.float32)
    
    # Train decoder parameters
    print("\nTraining decoder parameters...")
    tuner.train(
        X_train=X_train,
        y_train=y_train,
        epochs=50,
        batch_size=1  # Small batch size due to limited data
    )
    
    # Plot training history
    tuner.plot_training_history()
    
    # Test with a new case after tuning
    test_case = "Moderate fever with cough and fatigue"
    print(f"\nTesting tuned decoder with: {test_case}")
    
    visualizer.visualize_decoder_patterns(
        matrix,
        test_case,
        save_path="tuned_decoder_test"
    )

def test_decoder():
    """Test the MATRIX decoder's dual-path processing"""
    print("\nTesting MATRIX Decoder...")
    
    matrix = MATRIX()
    
    # Test case with clear severity indicators
    test_case = {
        "initial_responses": [
            {
                "question": "What is the patient's age?",
                "answer": "45",
                "type": "NUM"
            },
            {
                "question": "Please describe what brings you here today",
                "answer": "Severe chest pain with shortness of breath",
                "type": "FREE"
            }
        ],
        "followup_responses": [
            {
                "question": "When did the pain start?",
                "answer": "About an hour ago, suddenly",
                "type": "FREE"
            }
        ],
        "current_question": "Is the pain radiating anywhere?"
    }
    
    # Get state embedding
    state_embedding = matrix.state_encoder.encode_state(
        test_case["initial_responses"],
        test_case["followup_responses"],
        test_case["current_question"]
    )
    
    # Get agent perspectives
    optimist_view = matrix.optimist.evaluate(state_embedding, test_case["initial_responses"][1]["answer"])
    pessimist_view = matrix.pessimist.evaluate(state_embedding, test_case["initial_responses"][1]["answer"])
    
    # Process through decoder
    decoder_output = matrix.meta_learner.decoder(state_embedding)
    
    print("\nDecoder Outputs:")
    print("Optimist Path Output Shape:", decoder_output["optimist_output"].shape)
    print("Pessimist Path Output Shape:", decoder_output["pessimist_output"].shape)
    print("Combined Output Shape:", decoder_output["combined_output"].shape)
    
    # Show confidence scores
    print("\nConfidence Scores:")
    print(f"Optimist Confidence: {optimist_view['confidence']:.3f}")
    print(f"Pessimist Confidence: {pessimist_view['confidence']:.3f}")
    
    # Show attention weights
    opt_attention = torch.mean(decoder_output["optimist_output"], dim=1)
    pes_attention = torch.mean(decoder_output["pessimist_output"], dim=1)
    
    print("\nAttention Analysis:")
    print("Optimist Path Focus:", "Higher" if torch.mean(opt_attention) > torch.mean(pes_attention) else "Lower")
    print("Pessimist Path Focus:", "Higher" if torch.mean(pes_attention) > torch.mean(opt_attention) else "Lower")

def main():
    load_dotenv()
    
    print("Starting Enhanced MATRIX System Tests...")
    
    # Test pattern analysis with different scenarios
    test_pattern_analysis()
    
    # Test individual agents
    test_agents()
    
    # Test full pipeline with different cases
    test_matrix_pipeline()
    
    print("\nTests completed.")

if __name__ == "__main__":
    main()