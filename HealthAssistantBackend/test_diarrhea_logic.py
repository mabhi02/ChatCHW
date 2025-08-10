#!/usr/bin/env python3
"""
Test script to verify diarrhea-specific questioning logic.
This tests the hardcoded diarrhea questions and detection logic.
"""

def test_diarrhea_detection():
    """Test that diarrhea cases are properly detected."""
    print("Testing diarrhea detection logic...")
    
    # Test cases
    test_cases = [
        "My child has diarrhea",
        "I have diarrhoea", 
        "Patient has loose stools",
        "Child with diarrhea and fever",
        "No diarrhea symptoms",
        "Just a headache",
        "Fever and cough"
    ]
    
    diarrhea_terms = ['diarrhea', 'diarrhoea', 'loose stools']
    
    for test_case in test_cases:
        is_diarrhea = any(term in test_case.lower() for term in diarrhea_terms)
        print(f"'{test_case}' -> Diarrhea: {is_diarrhea}")
    
    print()

def test_diarrhea_questions():
    """Test that diarrhea questions are properly structured."""
    print("Testing diarrhea question structure...")
    
    # Mock the diarrhea questions function
    diarrhea_questions = [
        {
            "question": "How many days has the child had diarrhea?",
            "type": "MC", 
            "priority": 1,
            "options": [
                {"id": "1", "text": "Less than 14 days"},
                {"id": "2", "text": "14 days or more"},
                {"id": "3", "text": "Not sure about the exact duration"}
            ]
        },
        {
            "question": "Is the child able to drink or breastfeed normally?",
            "type": "MC",
            "priority": 2,
            "options": [
                {"id": "1", "text": "Yes, child drinks normally"},
                {"id": "2", "text": "No, child vomits everything"},
                {"id": "3", "text": "Child drinks slowly or reluctantly"},
                {"id": "4", "text": "Not sure"}
            ]
        }
    ]
    
    print(f"Total diarrhea questions: {len(diarrhea_questions)}")
    
    for i, question in enumerate(diarrhea_questions, 1):
        print(f"\nQuestion {i}: {question['question']}")
        print(f"Type: {question['type']}")
        print(f"Priority: {question['priority']}")
        print("Options:")
        for option in question['options']:
            print(f"  {option['id']}. {option['text']}")
    
    print()

def test_question_completion_check():
    """Test the logic for checking if all diarrhea questions are completed."""
    print("Testing question completion check logic...")
    
    # Mock session data with different completion states
    diarrhea_questions = [
        "How many days has the child had diarrhea?",
        "Is the child able to drink or breastfeed normally?"
    ]
    
    # Test case 1: No questions answered
    session_data_1 = {'smart_responses': []}
    answered_1 = set()
    for response in session_data_1['smart_responses']:
        for question in diarrhea_questions:
            if response['question'] == question:
                answered_1.add(question)
    
    remaining_1 = len(diarrhea_questions) - len(answered_1)
    print(f"Case 1 - No questions answered: {remaining_1} remaining (should be 2)")
    
    # Test case 2: Some questions answered
    session_data_2 = {
        'smart_responses': [
            {'question': 'How many days has the child had diarrhea?', 'answer': '3 days'}
        ]
    }
    answered_2 = set()
    for response in session_data_2['smart_responses']:
        for question in diarrhea_questions:
            if response['question'] == question:
                answered_2.add(question)
    
    remaining_2 = len(diarrhea_questions) - len(answered_2)
    print(f"Case 2 - Some questions answered: {remaining_2} remaining (should be 1)")
    
    # Test case 3: All questions answered
    session_data_3 = {
        'smart_responses': [
            {'question': 'How many days has the child had diarrhea?', 'answer': '3 days'},
            {'question': 'Is the child able to drink or breastfeed normally?', 'answer': 'Normal drinking'}
        ]
    }
    answered_3 = set()
    for response in session_data_3['smart_responses']:
        for question in diarrhea_questions:
            if response['question'] == question:
                answered_3.add(question)
    
    remaining_3 = len(diarrhea_questions) - len(answered_3)
    print(f"Case 3 - All questions answered: {remaining_3} remaining (should be 0)")
    
    print()

if __name__ == "__main__":
    print("=== Testing Diarrhea Questioning Logic ===\n")
    
    test_diarrhea_detection()
    test_diarrhea_questions()
    test_question_completion_check()
    
    print("=== All tests completed ===")
