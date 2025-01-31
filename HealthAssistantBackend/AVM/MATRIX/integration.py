from typing import Dict, List, Any
from .matrix_core import MATRIX
from .state_encoder import get_embedding
from .cmdML import vectorQuotes  # Import from your existing code

from .matrix_core import MATRIX
from .state_encoder import StateSpaceEncoder

from .cmdML import (
    get_initial_responses,
    get_followup_questions,
    get_followup_exams,
    get_embedding,
    vectorQuotes
)

class MATRIXWrapper:
    def __init__(self):
        self.matrix = MATRIX()
    
    def run_diagnosis(self):
        initial_responses = get_initial_responses()
        followup_responses = get_followup_questions(initial_responses)
        exam_responses = get_followup_exams(initial_responses, followup_responses)
        
        # Process through MATRIX
        state = self.matrix.process_state(
            initial_responses,
            followup_responses,
            exam_responses
        )
        
        return state

class MATRIXIntegration:
    def __init__(self):
        self.matrix = MATRIX()
        
    def process_followup_question(self,
                                initial_responses: List[Dict],
                                followup_responses: List[Dict],
                                current_question: str) -> bool:
        """Replace existing judge() function with MATRIX-based decision"""
        matrix_output = self.matrix.process_state(
            initial_responses,
            followup_responses,
            current_question
        )
        
        # Update Thompson sampling based on user feedback
        # (You'll need to add feedback collection)
        success = self._get_question_feedback()
        self.matrix.thompson_sampler.update(
            matrix_output["selected_agent"],
            success
        )
        
        return matrix_output["confidence"] > self.matrix.similarity_threshold
    
    def process_examination(self,
                          initial_responses: List[Dict],
                          followup_responses: List[Dict],
                          exam_responses: List[Dict],
                          current_exam: str) -> bool:
        """Replace existing judge_exam() with MATRIX-based decision"""
        matrix_output = self.matrix.process_state(
            initial_responses,
            followup_responses + exam_responses,
            current_exam
        )
        
        return (matrix_output["confidence"] > self.matrix.exam_similarity_threshold 
                or len(exam_responses) >= 5)