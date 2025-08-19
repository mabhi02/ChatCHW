import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .learnable_matrix import LearnableMATRIX

class MATRIXTrainingPipeline:
    """
    End-to-end training pipeline for the learnable MATRIX system.
    """
    
    def __init__(self, vector_store_index=None):
        self.vector_store_index = vector_store_index
        self.matrix = LearnableMATRIX()
        self.training_history = []
        
    def prepare_training_data(self, data_csv_path: str) -> List[Dict]:
        """
        Prepare training data from CSV with ground truth labels.
        """
        # Load data
        df = pd.read_csv(data_csv_path)
        
        training_cases = []
        
        for idx, row in df.iterrows():
            # Extract Q&A pairs from CHW Questions column
            chw_questions = row['CHW Questions']
            qa_pairs = self._parse_chw_questions(chw_questions)
            
            # Determine ground truth severity
            diagnosis = row['Diagnosis & Treatment (right answer for GraderBot.  Do NOT pass to the PatientBot LLM)']
            is_severe = self._is_severe_diagnosis(diagnosis)
            
            # Create training cases for each Q&A step
            initial_responses = [{"question": "complaint", "answer": row['Complaint']}]
            followup_responses = []
            
            for i, (question, answer) in enumerate(qa_pairs):
                followup_responses.append({"question": question, "answer": answer})
                
                # Determine if we should have stopped here
                should_have_stopped = self._should_have_stopped_here(
                    followup_responses, is_severe, i
                )
                
                # Create training case
                case = {
                    "initial_responses": initial_responses,
                    "followup_responses": followup_responses.copy(),
                    "current_question": question,
                    "ground_truth": {
                        "is_severe": is_severe,
                        "should_have_stopped": should_have_stopped,
                        "severity_score": self._calculate_severity_score(diagnosis),
                        "questions_asked": i + 1
                    }
                }
                
                training_cases.append(case)
        
        return training_cases
    
    def _parse_chw_questions(self, chw_questions: str) -> List[tuple]:
        """Parse CHW Questions column into Q&A pairs"""
        qa_pairs = []
        
        # Split by semicolon and parse each part
        parts = chw_questions.split(';')
        
        for part in parts:
            part = part.strip()
            if ':' in part:
                question, answer = part.split(':', 1)
                qa_pairs.append((question.strip(), answer.strip()))
        
        return qa_pairs
    
    def _is_severe_diagnosis(self, diagnosis: str) -> bool:
        """Determine if diagnosis indicates severe case"""
        severe_keywords = [
            "severe", "urgent", "emergency", "critical", "life-threatening",
            "immediate", "referral", "antibiotics", "hospital"
        ]
        
        diagnosis_lower = diagnosis.lower()
        return any(keyword in diagnosis_lower for keyword in severe_keywords)
    
    def _should_have_stopped_here(self, 
                                 followup_responses: List[Dict], 
                                 is_severe: bool, 
                                 question_index: int) -> bool:
        """
        Determine if the system should have stopped asking questions at this point.
        This is a heuristic based on medical knowledge.
        """
        # If we have enough information to make a confident decision
        if question_index >= 3:  # After 3 questions, we should have enough info
            return True
        
        # If we have a clear severe indicator
        for resp in followup_responses:
            answer = resp["answer"].lower()
            if any(term in answer for term in ["difficulty breathing", "unconscious", "unable to drink"]):
                return True
        
        return False
    
    def _calculate_severity_score(self, diagnosis: str) -> float:
        """Calculate severity score from diagnosis"""
        if self._is_severe_diagnosis(diagnosis):
            return 0.8
        else:
            return 0.2
    
    def train(self, 
              training_cases: List[Dict],
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the MATRIX system end-to-end.
        """
        # Split data
        train_cases, val_cases = train_test_split(
            training_cases, test_size=validation_split, random_state=42
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_efficiency": [],
            "val_efficiency": []
        }
        
        print(f"Training on {len(train_cases)} cases, validating on {len(val_cases)} cases")
        
        for epoch in range(epochs):
            # Training
            self.matrix.train()
            train_metrics = self._train_epoch(train_cases, batch_size)
            
            # Validation
            self.matrix.eval()
            val_metrics = self._validate_epoch(val_cases, batch_size)
            
            # Record history
            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["total_loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["train_efficiency"].append(train_metrics["efficiency"])
            history["val_efficiency"].append(val_metrics["efficiency"])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_metrics['total_loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.3f}, "
                      f"Val Acc: {val_metrics['accuracy']:.3f}")
        
        return history
    
    def _train_epoch(self, train_cases: List[Dict], batch_size: int) -> Dict[str, float]:
        """Train for one epoch"""
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        total_efficiency = 0.0
        
        # Shuffle cases
        np.random.shuffle(train_cases)
        
        # Process in batches
        for i in range(0, len(train_cases), batch_size):
            batch = train_cases[i:i+batch_size]
            
            # Training step
            metrics = self.matrix.train_step(batch, self.vector_store_index)
            total_loss += metrics["total_loss"]
            
            # Calculate accuracy and efficiency
            for case in batch:
                prediction = self.matrix.predict(
                    case["initial_responses"],
                    case["followup_responses"],
                    case["current_question"],
                    self.vector_store_index
                )
                
                # Accuracy
                if prediction["is_severe"] == case["ground_truth"]["is_severe"]:
                    correct_predictions += 1
                total_predictions += 1
                
                # Efficiency (questions saved)
                if prediction["should_stop"] == case["ground_truth"]["should_have_stopped"]:
                    total_efficiency += 1.0
        
        return {
            "total_loss": total_loss / (len(train_cases) // batch_size),
            "accuracy": correct_predictions / total_predictions,
            "efficiency": total_efficiency / len(train_cases)
        }
    
    def _validate_epoch(self, val_cases: List[Dict], batch_size: int) -> Dict[str, float]:
        """Validate for one epoch"""
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        total_efficiency = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_cases), batch_size):
                batch = val_cases[i:i+batch_size]
                
                batch_loss = 0.0
                for case in batch:
                    # Forward pass
                    predictions = self.matrix.forward(
                        case["initial_responses"],
                        case["followup_responses"],
                        case["current_question"],
                        self.vector_store_index
                    )
                    
                    # Compute loss
                    losses = self.matrix.compute_loss(predictions, case["ground_truth"])
                    case_loss = sum(losses.values())
                    batch_loss += case_loss.item()
                    
                    # Prediction
                    prediction = self.matrix.predict(
                        case["initial_responses"],
                        case["followup_responses"],
                        case["current_question"],
                        self.vector_store_index
                    )
                    
                    # Accuracy
                    if prediction["is_severe"] == case["ground_truth"]["is_severe"]:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Efficiency
                    if prediction["should_stop"] == case["ground_truth"]["should_have_stopped"]:
                        total_efficiency += 1.0
                
                total_loss += batch_loss / len(batch)
        
        return {
            "total_loss": total_loss / (len(val_cases) // batch_size),
            "accuracy": correct_predictions / total_predictions,
            "efficiency": total_efficiency / len(val_cases)
        }
    
    def evaluate(self, test_cases: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the trained model on test cases.
        """
        self.matrix.eval()
        
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "efficiency": 0.0,
            "avg_questions": 0.0,
            "early_stop_rate": 0.0
        }
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        total_questions = 0
        early_stops = 0
        
        with torch.no_grad():
            for case in test_cases:
                prediction = self.matrix.predict(
                    case["initial_responses"],
                    case["followup_responses"],
                    case["current_question"],
                    self.vector_store_index
                )
                
                # Classification metrics
                true_severe = case["ground_truth"]["is_severe"]
                pred_severe = prediction["is_severe"]
                
                if true_severe and pred_severe:
                    true_positives += 1
                elif not true_severe and pred_severe:
                    false_positives += 1
                elif true_severe and not pred_severe:
                    false_negatives += 1
                else:
                    true_negatives += 1
                
                # Efficiency metrics
                if prediction["should_stop"]:
                    early_stops += 1
                
                total_questions += case["ground_truth"]["questions_asked"]
        
        # Calculate metrics
        total_cases = len(test_cases)
        metrics["accuracy"] = (true_positives + true_negatives) / total_cases
        metrics["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        metrics["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
        metrics["efficiency"] = early_stops / total_cases
        metrics["avg_questions"] = total_questions / total_cases
        metrics["early_stop_rate"] = early_stops / total_cases
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.matrix.state_dict(),
            'optimizer_state_dict': self.matrix.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.matrix.load_state_dict(checkpoint['model_state_dict'])
        self.matrix.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
