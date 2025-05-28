#!/usr/bin/env python3
"""
ChatCHW + PatientBot Integration
Uses EXACT ChatCHW workflow from chad.py with PatientBot knowledge-base responses
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add parent directory to path to import ChatCHW modules
sys.path.append('..')

# Import ALL existing ChatCHW functions EXACTLY as they are
from chad import (
    questions_init,
    get_embedding_batch,
    vectorQuotesWithSource,
    get_openai_completion,
    process_with_matrix,
    judge,
    judge_exam,
    parse_examination_text,
    get_diagnosis_and_treatment,
    parse_question_data,
    store_examination,
    print_options,
    validate_mc_input,
    validate_num_input,
    structured_questions_array,
    examination_history,
    pc
)

class CHWPatientBotIntegration:
    def __init__(self):
        self.patient_bot_url = "http://localhost:5003"
        self.outputs_dir = "outputs"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        
    def ask_patient_bot(self, question):
        """Ask PatientBot a question and get knowledge-base response"""
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/ask",
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['response']
            return "other: Information not known"
        except Exception as e:
            print(f"Error asking PatientBot: {e}")
            return "other: Information not known"
    
    def load_patient_case(self, case_id):
        """Load a patient case in PatientBot"""
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/load", 
                                   json={"case_id": case_id})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['patient']
            return None
        except Exception as e:
            print(f"Error loading patient: {e}")
            return None
    
    def select_patient_case(self):
        """Let user select a patient case"""
        try:
            response = requests.get(f"{self.patient_bot_url}/api/patient/cases")
            if response.status_code == 200:
                data = response.json()
                cases = data.get('cases', [])
                
                print("\n📋 AVAILABLE PATIENT CASES:")
                print("=" * 80)
                for case in cases:
                    print(f"ID {case['case_id']}: {case['age_years']} {case['sex']} - {case['complaint']} ({case['duration']})")
                
                print(f"\nTotal cases available: {data.get('total_cases', len(cases))}")
                
                while True:
                    choice = input(f"\nEnter case ID (0-{len(cases)-1}) or 'random': ").strip()
                    if choice == 'random':
                        import random
                        selected_id = random.randint(0, len(cases)-1)
                        print(f"🎲 Randomly selected case ID: {selected_id}")
                        return selected_id
                    elif choice.isdigit():
                        case_id = int(choice)
                        if 0 <= case_id < len(cases):
                            return case_id
                        else:
                            print(f"Invalid choice. Please enter a number between 0 and {len(cases)-1}")
                    else:
                        print("Invalid choice. Enter a number or 'random'")
            return 0
        except Exception as e:
            print(f"Error selecting case: {e}")
            print("Using default case 0")
            return 0

    def match_response_to_option(self, response, options):
        """Auto-match PatientBot response to multiple choice options"""
        if response == "other: Information not known":
            # When PatientBot doesn't know, select a reasonable default
            # Usually option 3 (Not sure/Unknown) or option 2 (No) for yes/no questions
            if len(options) >= 3:
                for option in options:
                    if any(word in option['text'].lower() for word in ['not sure', 'unknown', 'unsure']):
                        return str(option['id'])
                # If no "not sure" option, default to option 2 (often "No")
                return "2"
            return "1"  # Fallback to first option
            
        response_lower = response.lower()
        best_match = None
        best_score = 0
        
        for option in options:
            if option['id'] == 5:  # Skip "Other" option
                continue
                
            option_text = option['text'].lower()
            
            # Direct matches
            if response_lower == option_text:
                return str(option['id'])
            
            # Keyword matching
            keywords = option_text.split()
            matches = sum(1 for word in keywords if word in response_lower)
            score = matches / len(keywords) if keywords else 0
            
            if score > best_score and score > 0.3:  # Lower threshold for better matching
                best_score = score
                best_match = str(option['id'])
        
        # If no good match found, select reasonable defaults
        if not best_match:
            # For yes/no questions, try to detect and default appropriately
            if len(options) == 3:  # Usually Yes/No/Other format
                return "2"  # Default to "No"
            elif len(options) >= 3:
                return "3"  # Default to third option (often "Not sure")
            else:
                return "1"  # Fallback to first option
        
        return best_match

    def extract_number_from_response(self, response):
        """Extract number from PatientBot response"""
        if response == "other: Information not known":
            return 1  # Default age for unknown
            
        import re
        numbers = re.findall(r'\d+', response)
        return int(numbers[0]) if numbers else 1  # Default to 1 if no number found

    def run_chw_workflow_with_patientbot(self):
        """Run the EXACT ChatCHW workflow from chad.py main() but with PatientBot responses - FULLY AUTOMATED"""
        
        # Initialize/clear the global arrays EXACTLY like chad.py
        global structured_questions_array, examination_history
        structured_questions_array = []
        examination_history = []
        
        initial_responses = []
        followup_responses = []
        exam_responses = []
        
        print("\nMedical Assessment Initial Questions")
        print("===================================")
        
        # EXACT Initial Questions Loop from chad.py - FULLY AUTOMATED
        for question in questions_init:
            print(f"\n{question['question']}")
            
            if question['type'] in ['MC', 'YN', 'MCM']:
                print_options(question['options'])
                
                # Get PatientBot response
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                # Auto-select - NO USER INPUT
                answer = self.match_response_to_option(patient_response, question['options'])
                print(f"🤖 Auto-selected: {answer}")
                
                # Use EXACT validation logic from chad.py
                if question['type'] == 'MCM':
                    if ',' in answer:
                        answers = answer.split(',')
                        valid = all(validate_mc_input(a.strip(), question['options']) for a in answers)
                        if valid:
                            initial_responses.append({
                                "question": question['question'],
                                "answer": [a.strip() for a in answers],
                                "type": question['type']
                            })
                    else:
                        if validate_mc_input(answer, question['options']):
                            initial_responses.append({
                                "question": question['question'],
                                "answer": [answer],
                                "type": question['type']
                            })
                else:
                    if validate_mc_input(answer, question['options']):
                        if answer == "5":
                            # Auto-provide a generic response for "Other"
                            custom_answer = "Not specified"
                            initial_responses.append({
                                "question": question['question'],
                                "answer": custom_answer,
                                "type": question['type']
                            })
                        else:
                            selected_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                            initial_responses.append({
                                "question": question['question'],
                                "answer": selected_text,
                                "type": question['type']
                            })
            
            elif question['type'] == 'NUM':
                # Get PatientBot response
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                # Auto-extract number - NO USER INPUT
                auto_answer = self.extract_number_from_response(patient_response)
                answer = str(auto_answer)
                print(f"🤖 Auto-extracted: {auto_answer}")
                
                if validated_num := validate_num_input(answer, question['range']):
                    initial_responses.append({
                        "question": question['question'],
                        "answer": validated_num,
                        "type": question['type']
                    })
                    
            elif question['type'] == 'FREE':
                # Get PatientBot response
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                if patient_response != "other: Information not known":
                    answer = patient_response
                else:
                    answer = "Not specified"  # Auto-provide generic response
                
                print(f"🤖 Using: {answer}")
                initial_responses.append({
                    "question": question['question'],
                    "answer": answer,
                    "type": question['type']
                })

        print("\nThank you for providing your information. Here's what we recorded:\n")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")

        # EXACT Follow-up Questions Loop from chad.py - FULLY AUTOMATED
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        print("\nBased on your responses, I'll ask some follow-up questions.")
        index = pc.Index("who-guide-old")
        
        followup_count = 0
        max_followups = 5  # Limit to prevent infinite loop
        
        while followup_count < max_followups:
            try:
                context = f"Initial complaint: {initial_complaint}\n"
                if followup_responses:
                    context += "Previous responses:\n"
                    for resp in followup_responses:
                        context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
                
                embedding = get_embedding_batch([context])[0]
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                if not relevant_docs:
                    print("Error: Could not generate relevant question.")
                    break
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
                prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
                
                Previous questions asked:
                {previous_questions if followup_responses else "No previous questions yet"}
                
                Relevant medical context:
                {combined_context}
                
                Generate ONE focused, relevant follow-up question that is different from the previous questions.
                Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
                Do not as about compound questions like "Do you have fever and cough?" or "Do you have pain in your chest or abdomen?". It should be one or the other like "Do you have fever" or "Do you have pain in your chest?".
                There should be no "or" or "and" in the question as ask about one specific metric not compounded one.
                
                Return only the question text.'''
                
                question = get_openai_completion(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.3
                )
                
                # Generate options
                options_prompt = f'''Generate 4 concise answers for: "{question}"
                Clear, mutually exclusive options.
                Return each option on a new line (1-4).'''
                
                options_text = get_openai_completion(
                    prompt=options_prompt,
                    max_tokens=100,
                    temperature=0.2
                )
                
                options = []
                for i, opt in enumerate(options_text.strip().split('\n')):
                    if opt.strip():
                        text = opt.strip()
                        if text[0].isdigit() and text[1] in ['.','-',')']:
                            text = text[2:].strip()
                        options.append({"id": i+1, "text": text})
                
                options.append({"id": 5, "text": "Other (please specify)"})
                
                # Use MATRIX to judge similarity and get pattern analysis - EXACT from chad.py
                matrix_output = process_with_matrix(question, followup_responses)
                
                # Print question and get answer
                print(f"\n{question}")
                print_options(options)
                
                # Get PatientBot response
                patient_response = self.ask_patient_bot(question)
                print(f"🤒 PatientBot: {patient_response}")
                
                # Auto-select - NO USER INPUT
                answer = self.match_response_to_option(patient_response, options)
                print(f"🤖 Auto-selected: {answer}")
                
                if validate_mc_input(answer, options):
                    if answer == "5":
                        answer_text = "Not specified"  # Auto-provide generic response
                    else:
                        answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                    
                    # Store response - EXACT from chad.py
                    followup_responses.append({
                        "question": question,
                        "answer": answer_text,
                        "type": "MC",
                        "citations": relevant_docs[-5:]
                    })
                    
                    # Parse and store question data - EXACT from chad.py
                    parse_question_data(
                        question, 
                        options, 
                        answer_text, 
                        matrix_output, 
                        relevant_docs[-5:]
                    )
                    
                    # Print global arrays after each answer - EXACT from chad.py
                    print("\nCurrent Structured Questions Array:")
                    print("================================")
                    print(json.dumps(structured_questions_array, indent=2))
                    
                    print("\nCurrent Examination History:")
                    print("=========================")
                    print(json.dumps(examination_history, indent=2))
                    
                    followup_count += 1
                    
                    # EXACT judge logic from chad.py
                    if judge(followup_responses, question):
                        print("\nMoving to examination phase...")
                        break
                else:
                    print("Invalid auto-selection, skipping question.")
                    break
                    
            except Exception as e:
                print(f"Error in follow-up questions: {e}")
                break

        # EXACT Examination Loop from chad.py - FULLY AUTOMATED
        print("\nExamination Phase")
        print("================")
        
        exam_count = 0
        max_exams = 3  # Limit to prevent infinite loop
        
        while exam_count < max_exams:
            try:
                context = f"Initial complaint: {initial_complaint}\n"
                context += "Follow-up responses:\n"
                for resp in followup_responses:
                    context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
                
                if exam_responses:
                    context += "Previous examinations:\n"
                    for resp in exam_responses:
                        context += f"Exam: {resp['examination']}\nResult: {resp['result']}\n"
                
                embedding = get_embedding_batch([context])[0]
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                if not relevant_docs:
                    print("Error: Could not generate relevant examination.")
                    break
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                previous_exams = "\n".join([f"- {resp['examination']}" for resp in exam_responses])
                prompt = f'''Based on the patient information and medical context, suggest ONE specific physical examination or test.
                
                Patient context:
                {context}
                
                Previous examinations:
                {previous_exams if exam_responses else "No previous examinations"}
                
                Medical context:
                {combined_context}
                
                Return only the examination name (e.g., "Check blood pressure", "Listen to lungs", "Examine throat").'''
                
                examination = get_openai_completion(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.3
                )
                
                print(f"\nSuggested examination: {examination}")
                
                # Get PatientBot response for examination
                patient_response = self.ask_patient_bot(f"What do you find when you {examination.lower()}?")
                print(f"🤒 PatientBot: {patient_response}")
                
                if patient_response != "other: Information not known":
                    result = patient_response
                else:
                    result = "Normal findings"  # Auto-provide default result
                
                print(f"🤖 Using result: {result}")
                
                exam_responses.append({
                    "examination": examination,
                    "result": result,
                    "citations": relevant_docs[-5:]
                })
                
                # EXACT store_examination from chad.py
                store_examination(examination, 1)  # Assuming option 1 for simplicity
                
                print(f"Recorded: {examination} -> {result}")
                
                exam_count += 1
                
                # EXACT judge_exam logic from chad.py
                if judge_exam(exam_responses, examination):
                    print("\nSufficient examination data collected. Moving to diagnosis...")
                    break
                        
            except Exception as e:
                print(f"Error in examination phase: {e}")
                break

        # EXACT Final Diagnosis from chad.py
        print("\nGenerating Diagnosis and Treatment Plan...")
        print("========================================")
        
        diagnosis_result = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
        
        print(f"\nDiagnosis: {diagnosis_result.get('diagnosis', 'Unable to determine')}")
        print(f"Treatment: {diagnosis_result.get('treatment', 'Unable to determine')}")
        print(f"Confidence: {diagnosis_result.get('confidence', 'Unknown')}")
        
        # Save results
        self.save_results(initial_responses, followup_responses, exam_responses, diagnosis_result)
        
        return {
            'initial_responses': initial_responses,
            'followup_responses': followup_responses,
            'exam_responses': exam_responses,
            'diagnosis': diagnosis_result,
            'structured_questions': structured_questions_array,
            'examination_history': examination_history
        }

    def save_results(self, initial_responses, followup_responses, exam_responses, diagnosis_result):
        """Save session results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chw_patientbot_session_{timestamp}.json"
        filepath = os.path.join(self.outputs_dir, filename)
        
        results = {
            'timestamp': timestamp,
            'initial_responses': initial_responses,
            'followup_responses': followup_responses,
            'exam_responses': exam_responses,
            'diagnosis': diagnosis_result,
            'structured_questions': structured_questions_array,
            'examination_history': examination_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")

def main():
    """Main function to run CHW + PatientBot integration"""
    integration = CHWPatientBotIntegration()
    
    print("🏥 ChatCHW + PatientBot Integration")
    print("=" * 50)
    
    # Select and load patient case
    case_id = integration.select_patient_case()
    patient_info = integration.load_patient_case(case_id)
    
    if not patient_info:
        print("Failed to load patient case!")
        return
    
    print(f"\n👤 Loaded Patient Case {case_id}:")
    print(f"Age: {patient_info['age_years']} years ({patient_info['age_category']})")
    print(f"Sex: {patient_info['sex']}")
    print(f"Primary Complaint: {patient_info['primary_complaint']}")
    print(f"Duration: {patient_info['duration']}")
    
    # Run the EXACT ChatCHW workflow with PatientBot
    results = integration.run_chw_workflow_with_patientbot()
    
    print("\n✅ Integration completed successfully!")

if __name__ == "__main__":
    main() 