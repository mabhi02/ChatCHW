from matrix_core import MATRIX
from integration import MATRIXIntegration

from typing import Dict, List, Any
from .cmdML import (
    get_initial_responses,
    get_followup_questions,
    get_followup_exams
)

def main():
    # Initialize MATRIX system
    matrix_integration = MATRIXIntegration()
    
    try:
        # Your existing main() function flow but using MATRIX
        initial_responses = get_initial_responses()
        
        # Use MATRIX for followup questions
        followup_responses = get_followup_questions(
            initial_responses,
            matrix_integration
        )
        
        # Use MATRIX for examinations
        exam_responses = get_followup_exams(
            initial_responses,
            followup_responses,
            matrix_integration
        )
        
        # Rest of your existing logic for diagnosis and treatment
        
    except Exception as e:
        print(f"Error in MATRIX execution: {e}")
        raise

if __name__ == "__main__":
    main()