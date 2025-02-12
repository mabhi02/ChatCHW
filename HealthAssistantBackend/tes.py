from ML import create_assessment

def test_medical_assessment():
    # Create a new assessment instance
    assessment = create_assessment()
    
    # 1. Test initial questions
    print("\nTesting Initial Questions...")
    initial_questions = assessment.get_initial_questions()
    print(f"Got {len(initial_questions)} initial questions")
    
    # Submit some test responses
    test_responses = [
        {
            "question": "What is the patient's sex?",
            "answer": "Male",
            "type": "MC"
        },
        {
            "question": "What is the patient's age?",
            "answer": 45,
            "type": "NUM"
        },
        {
            "question": "Does the patient have a caregiver?",
            "answer": "No",
            "type": "MC"
        },
        {
            "question": "Who is accompanying the patient?",
            "answer": ["None"],
            "type": "MCM"
        },
        {
            "question": "Please describe what brings you here today",
            "answer": "I've been having severe headaches for the past week with some nausea",
            "type": "FREE"
        }
    ]
    
    for response in test_responses:
        result = assessment.submit_initial_response(response["question"], response["answer"])
        print(f"\nSubmitted initial response: {result}")
    
    # 2. Test followup questions
    print("\nTesting Followup Questions...")
    followup_question = assessment.generate_followup_question()
    print(f"\nGenerated followup question: {followup_question}")
    
    if followup_question:
        # Submit a test followup response
        followup_response = assessment.submit_followup_response(
            followup_question,
            followup_question["options"][0]["text"]  # Select first option
        )
        print(f"\nSubmitted followup response: {followup_response}")
    
    # 3. Test examination generation
    print("\nTesting Examination Generation...")
    examination = assessment.generate_examination()
    print(f"\nGenerated examination: {examination}")
    
    if examination:
        # Submit a test examination result
        exam_result = assessment.submit_examination_result(
            examination,
            examination["options"][0]["text"]  # Select first option
        )
        print(f"\nSubmitted examination result: {exam_result}")
    
    # 4. Test diagnosis generation
    print("\nTesting Diagnosis Generation...")
    diagnosis = assessment.generate_diagnosis_and_treatment()
    print(f"\nGenerated diagnosis and treatment: {diagnosis}")
    
    # 5. Test state management
    print("\nTesting State Management...")
    state = assessment.get_state()
    print("\nFinal state:")
    print(f"- Initial responses: {len(state['initial_responses'])}")
    print(f"- Followup responses: {len(state['followup_responses'])}")
    print(f"- Exam responses: {len(state['exam_responses'])}")
    print(f"- Structured questions: {len(state['structured_questions'])}")
    
    # Test save/load
    assessment.save_state("test_state.json")
    print("\nSaved state to test_state.json")
    
    new_assessment = create_assessment()
    new_assessment.load_state("test_state.json")
    print("Loaded state successfully")
    
    return assessment

if __name__ == "__main__":
    try:
        test_medical_assessment()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")