#!/usr/bin/env python3
"""
Test script to verify diarrhea-specific fixes in the prompts.
This script tests that:
1. Follow-up questions for diarrhea cases use the correct format
2. Treatment plans for diarrhea cases include zinc
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompts_infer import (
    get_main_followup_question_prompt,
    get_diagnosis_prompt,
    get_treatment_prompt
)

def test_diarrhea_followup_question():
    """Test that diarrhea cases generate the correct follow-up question format."""
    print("Testing diarrhea follow-up question format...")
    
    # Test with diarrhea complaint
    diarrhea_complaint = "My child has diarrhea"
    previous_questions = "What is your child's age?"
    medical_context = "Diarrhea is a common condition in children..."
    
    prompt = get_main_followup_question_prompt(
        initial_complaint=diarrhea_complaint,
        previous_questions=previous_questions,
        combined_context=medical_context
    )
    
    print("Generated prompt contains diarrhea instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt)
    print("Generated prompt contains correct format:", "For how long has the child had diarrhoea? 1. 3 days 2. 14 days or more 3. 7 days or more 4. Not specified 5. Other (please specify)" in prompt)
    
    # Test with non-diarrhea complaint
    non_diarrhea_complaint = "My child has fever"
    prompt2 = get_main_followup_question_prompt(
        initial_complaint=non_diarrhea_complaint,
        previous_questions=previous_questions,
        combined_context=medical_context
    )
    
    print("Non-diarrhea prompt contains diarrhea instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt2)
    print()

def test_diarrhea_diagnosis():
    """Test that diarrhea cases include zinc in diagnosis."""
    print("Testing diarrhea diagnosis zinc inclusion...")
    
    # Test with diarrhea case
    diarrhea_complaint = "My child has diarrhea"
    symptoms_text = "Child has loose stools for 3 days"
    exam_results = "No specific examination performed"
    medical_content = "Diarrhea treatment includes ORS..."
    
    prompt = get_diagnosis_prompt(
        initial_complaint=diarrhea_complaint,
        symptoms_text=symptoms_text,
        exam_results_text=exam_results,
        medical_guide_content=medical_content
    )
    
    print("Diarrhea diagnosis prompt contains zinc instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt)
    print("Diarrhea diagnosis prompt mentions zinc:", "zinc supplementation" in prompt)
    print("Diarrhea diagnosis prompt mentions WHO 2012 guidelines:", "WHO 2012 guidelines" in prompt)
    
    # Test with non-diarrhea case
    non_diarrhea_complaint = "My child has fever"
    non_diarrhea_symptoms = "Child has high temperature"
    
    prompt2 = get_diagnosis_prompt(
        initial_complaint=non_diarrhea_complaint,
        symptoms_text=non_diarrhea_symptoms,
        exam_results_text=exam_results,
        medical_guide_content=medical_content
    )
    
    print("Non-diarrhea diagnosis prompt contains zinc instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt2)
    print()

def test_diarrhea_treatment():
    """Test that diarrhea cases include zinc in treatment."""
    print("Testing diarrhea treatment zinc inclusion...")
    
    # Test with diarrhea case
    diarrhea_complaint = "My child has diarrhea"
    symptoms_text = "Child has loose stools for 3 days"
    exam_results = "No specific examination performed"
    diagnosis = "Uncomplicated diarrhea"
    medical_content = "Diarrhea treatment includes ORS..."
    
    prompt = get_treatment_prompt(
        initial_complaint=diarrhea_complaint,
        symptoms_text=symptoms_text,
        exam_results_text=exam_results,
        diagnosis=diagnosis,
        medical_guide_content=medical_content
    )
    
    print("Diarrhea treatment prompt contains zinc instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt)
    print("Diarrhea treatment prompt mentions zinc:", "zinc supplementation" in prompt)
    print("Diarrhea treatment prompt mentions 10-14 days:", "10-14 days" in prompt)
    print("Diarrhea treatment prompt mentions both ORS and zinc:", "both ORS and zinc" in prompt)
    
    # Test with non-diarrhea case
    non_diarrhea_complaint = "My child has fever"
    non_diarrhea_symptoms = "Child has high temperature"
    non_diarrhea_diagnosis = "Fever of unknown origin"
    
    prompt2 = get_treatment_prompt(
        initial_complaint=non_diarrhea_complaint,
        symptoms_text=non_diarrhea_symptoms,
        exam_results_text=exam_results,
        diagnosis=non_diarrhea_diagnosis,
        medical_guide_content=medical_content
    )
    
    print("Non-diarrhea treatment prompt contains zinc instructions:", "SPECIAL INSTRUCTIONS FOR DIARRHEA CASES" in prompt2)
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING DIARRHEA-SPECIFIC FIXES")
    print("=" * 60)
    print()
    
    test_diarrhea_followup_question()
    test_diarrhea_diagnosis()
    test_diarrhea_treatment()
    
    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 