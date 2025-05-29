#!/usr/bin/env python3
"""
Simple Patient Case Runner
Pick a case and run it through CHW workflow
"""

import pandas as pd
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

# Import the EXACT examination generation functions from app.py
sys.path.append('../HealthAssistantBackend')
try:
    from app import generate_examination
    print("✅ Successfully imported generate_examination from app.py")
except ImportError as e:
    print(f"❌ Failed to import generate_examination from app.py: {e}")
    # Create a minimal fallback version
    def generate_examination(session_data):
        return {
            "status": "error",
            "output": "Examination generation not available"
        }

class SimplePatientRunner:
    def __init__(self):
        self.patient_bot_url = "http://localhost:5003"
        self.outputs_dir = "outputs"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
    
    def show_cases(self, limit=25):
        """Show available patient cases"""
        df = pd.read_csv('data.csv')
        
        print("📋 PATIENT CASES:")
        print("=" * 80)
        print(f"{'ID':<3} {'Age':<3} {'Sex':<6} {'Complaint':<25} {'Duration':<15}")
        print("-" * 80)
        
        for i in range(min(limit, len(df))):
            row = df.iloc[i]
            print(f"{i:<3} {row['age (years)']:<3} {row['Sex']:<6} {str(row['Complaint']):<25} {str(row['Duration']):<15}")
        
        print(f"\nShowing {min(limit, len(df))} of {len(df)} total cases")
        return df
    
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
            print(f"❌ Error loading patient: {e}")
            print("Make sure PatientBot API is running: python API.py")
            return None
    
    def ask_patient_bot(self, question):
        """Ask PatientBot a question"""
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/ask",
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['response']
            return "other: Information not known"
        except Exception as e:
            return "other: Information not known"
    
    def match_response_to_option(self, response, options, patient_info=None):
        """Auto-match PatientBot response to multiple choice options using improved AI"""
        
        # Handle "Information not known" - should always select option 5 (Other)
        if response == "other: Information not known":
            # Find option 5 or any "Other" option
            for option in options:
                if option['id'] == 5 or 'other' in option['text'].lower():
                    return str(option['id']), "No information found"
            # Fallback if no "Other" option exists
            return str(len(options)), "No information found"
        
        # Enhanced AI prompt with patient context
        options_text = "\n".join([f"{opt['id']}: {opt['text']}" for opt in options])
        
        patient_context = ""
        if patient_info:
            patient_context = f"""
Patient Case Context:
- Age: {patient_info.get('age_years', 'Unknown')} years
- Sex: {patient_info.get('sex', 'Unknown')}
- Primary Complaint: {patient_info.get('primary_complaint', 'Unknown')}
- Duration: {patient_info.get('duration', 'Unknown')}
"""
        
        prompt = f"""Given this patient response: "{response}"
{patient_context}
Available options:
{options_text}

CRITICAL RULES:
1. If patient says "Yes" and there are multiple Yes options (like options 1,2,3), pick the MOST SPECIFIC/APPROPRIATE one based on the patient's case
2. If patient says a duration like "2 days", match it to the closest duration option, NOT option 5
3. If patient says "Female/Male", pick the exact gender option, NOT option 5
4. Only use option 5 (Other) if the response truly doesn't match ANY of the specific options 1-4
5. Be SMART about medical context - use the patient's complaint/symptoms to pick the best match

Examples:
- Patient with "Cough" complaint + says "Yes" to breathing difficulty → pick the breathing-related yes option
- Patient says "2 days" → pick closest duration match (e.g., "3 days" option)
- Patient says "Female" → pick option 2 (Female), not option 5

Return ONLY the option number (e.g., "2")."""

        try:
            ai_choice = get_openai_completion(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            ).strip()
            
            # Extract just the number
            import re
            match = re.search(r'\d+', ai_choice)
            if match:
                choice_num = int(match.group())
                if 1 <= choice_num <= len(options):
                    # If option 5 selected, determine what specific answer to provide
                    if choice_num == 5:
                        if response == "other: Information not known":
                            specific_answer = "No information found"
                        else:
                            specific_answer = response
                    else:
                        specific_answer = None
                    return str(choice_num), specific_answer
        except Exception as e:
            print(f"AI matching error: {e}")
        
        # Fallback: return option 5 (Other) if available, otherwise first option
        for option in options:
            if option['id'] == 5:
                return "5", "No information found"
        return "1", None

    def extract_number_from_response(self, response):
        """Extract number from PatientBot response"""
        if response == "other: Information not known":
            return 1  # Default age for unknown
            
        import re
        numbers = re.findall(r'\d+', response)
        return int(numbers[0]) if numbers else 1

    def run_chw_workflow(self, patient_info=None):
        """Run the EXACT ChatCHW workflow - FULLY AUTOMATED"""
        
        # Initialize/clear the global arrays EXACTLY like chad.py
        global structured_questions_array, examination_history
        structured_questions_array = []
        examination_history = []
        
        initial_responses = []
        followup_responses = []
        exam_responses = []
        
        print("\n🏥 MEDICAL ASSESSMENT - INITIAL QUESTIONS")
        print("=" * 60)
        
        # EXACT Initial Questions Loop from chad.py - FULLY AUTOMATED
        for question in questions_init:
            print(f"\n{question['question']}")
            
            if question['type'] in ['MC', 'YN', 'MCM']:
                print_options(question['options'])
                
                # Get PatientBot response
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                # Auto-select - NO USER INPUT
                answer, specific_answer = self.match_response_to_option(patient_response, question['options'], patient_info)
                print(f"📋 Option Selection: {answer}")
                if specific_answer:
                    print(f"✏️ Specific Answer: {specific_answer}")
                else:
                    print(f"✏️ Specific Answer: None")
                
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
                            # Use the specific answer for option 5
                            custom_answer = specific_answer if specific_answer else "Not specified"
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
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                auto_answer = self.extract_number_from_response(patient_response)
                answer = str(auto_answer)
                print(f"📋 Extracted Number: {auto_answer}")
                print(f"✏️ Specific Answer: None")
                
                if validated_num := validate_num_input(answer, question['range']):
                    initial_responses.append({
                        "question": question['question'],
                        "answer": validated_num,
                        "type": question['type']
                    })
                    
            elif question['type'] == 'FREE':
                patient_response = self.ask_patient_bot(question['question'])
                print(f"🤒 PatientBot: {patient_response}")
                
                if patient_response != "other: Information not known":
                    answer = patient_response
                else:
                    # Generate intelligent answer from patient case data
                    if patient_info and 'what brings you here today' in question['question'].lower():
                        # Create complaint from patient case data
                        complaint_parts = []
                        if patient_info.get('primary_complaint'):
                            complaint_parts.append(patient_info['primary_complaint'])
                        if patient_info.get('duration'):
                            complaint_parts.append(f"for {patient_info['duration']}")
                        
                        if complaint_parts:
                            answer = " ".join(complaint_parts)
                        else:
                            answer = "Not specified"
                    else:
                        answer = "Not specified"
                
                print(f"📋 Generated Response: {answer}")
                print(f"✏️ Specific Answer: None")
                initial_responses.append({
                    "question": question['question'],
                    "answer": answer,
                    "type": question['type']
                })

        print("\n🏥 INITIAL ASSESSMENT COMPLETE")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")

        # EXACT Follow-up Questions Loop from chad.py - FULLY AUTOMATED
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        print("\n🔍 FOLLOW-UP QUESTIONS")
        print("=" * 40)
        index = pc.Index("who-guide-old")
        
        followup_count = 0
        max_followups = 5
        
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
                    print("No more relevant questions.")
                    break
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
                prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
                
                Previous questions asked:
                {previous_questions if followup_responses else "No previous questions yet"}
                
                Relevant medical context:
                {combined_context}
                
                Generate ONE focused, relevant follow-up question that is different from the previous questions.
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
                
                # Use MATRIX
                matrix_output = process_with_matrix(question, followup_responses)
                
                print(f"\n{question}")
                print_options(options)
                
                patient_response = self.ask_patient_bot(question)
                print(f"🤒 PatientBot: {patient_response}")
                
                answer, specific_answer = self.match_response_to_option(patient_response, options, patient_info)
                print(f"📋 Option Selection: {answer}")
                if specific_answer:
                    print(f"✏️ Specific Answer: {specific_answer}")
                else:
                    print(f"✏️ Specific Answer: None")
                
                if validate_mc_input(answer, options):
                    if answer == "5":
                        # Use the specific answer for option 5
                        answer_text = specific_answer if specific_answer else "Not specified"
                    else:
                        answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                    
                    followup_responses.append({
                        "question": question,
                        "answer": answer_text,
                        "type": "MC",
                        "citations": relevant_docs[-5:]
                    })
                    
                    parse_question_data(
                        question, 
                        options, 
                        answer_text, 
                        matrix_output, 
                        relevant_docs[-5:]
                    )
                    
                    followup_count += 1
                    
                    if judge(followup_responses, question):
                        print("\n✅ Moving to examination phase...")
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"Error in follow-up questions: {e}")
                break

        # EXACT Examination Loop from chad.py - FULLY AUTOMATED
        print("\n🔬 EXAMINATION PHASE")
        print("=" * 30)
        
        exam_count = 0
        max_exams = 3
        
        while exam_count < max_exams:
            try:
                # Create session data structure that generate_examination expects
                session_data = {
                    'session_id': f'run_case_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'initial_responses': initial_responses,
                    'followup_responses': followup_responses,
                    'exam_responses': exam_responses,
                    'phase': 'exam'
                }
                
                # Use the EXACT same generate_examination function from app.py
                examination_result = generate_examination(session_data)
                
                if examination_result.get('status') != 'success':
                    print(f"❌ Examination generation failed: {examination_result.get('message', 'Unknown error')}")
                    break
                
                # Get the examination details
                examination_output = examination_result['output']
                examination_metadata = examination_result.get('metadata', {})
                examination_options = examination_metadata.get('options', [])
                
                print(f"\n{examination_output}")
                
                # Handle different examination types
                exam_type = examination_metadata.get('type', 'EXAM')
                
                if exam_type == "NO_INFO_EXAM":
                    # For cases where medical guide has no examination info
                    print("📋 Please select an option:")
                    for i, option in enumerate(examination_options, 1):
                        print(f"  {i}: {option['text']}")
                    
                    # Auto-select option 1 (proceed with assessment)
                    print(f"🤒 PatientBot: Proceed with assessment")
                    print(f"📋 Option Selection: 1")
                    print(f"✏️ Specific Answer: None")
                    
                    exam_responses.append({
                        "examination": "No specific examination available in medical guide",
                        "result": "Proceeded with general assessment",
                        "type": "NO_INFO_EXAM"
                    })
                    
                    # Move directly to diagnosis
                    break
                
                else:
                    # Regular examination with findings options
                    if examination_options:
                        # Ask PatientBot about examination findings
                        # Extract examination name from the output
                        lines = examination_output.split('\n')
                        examination_name = "Physical examination"
                        for line in lines:
                            if "Examination:" in line or "Recommended Examination:" in line:
                                examination_name = line.split(':')[1].strip()
                                break
                        
                        # Ask PatientBot for findings
                        finding_question = f"During {examination_name.lower()}, what findings do you observe?"
                        patient_response = self.ask_patient_bot(finding_question)
                        print(f"🤒 PatientBot: {patient_response}")
                        
                        # Match the response to examination options
                        answer, specific_answer = self.match_response_to_option(
                            patient_response, examination_options, patient_info
                        )
                        
                        print(f"📋 Option Selection: {answer}")
                        if specific_answer:
                            print(f"✏️ Specific Answer: {specific_answer}")
                        else:
                            print(f"✏️ Specific Answer: None")
                        
                        # Get the selected finding text
                        if answer == "5" and specific_answer:
                            result_text = specific_answer
                        elif answer.isdigit() and int(answer) <= len(examination_options):
                            selected_option = next(
                                (opt for opt in examination_options if str(opt['id']) == answer), 
                                examination_options[0]
                            )
                            result_text = selected_option['text']
                        else:
                            result_text = "Normal findings"
                        
                        # Store the examination result
                        exam_responses.append({
                            "examination": examination_name,
                            "result": result_text,
                            "citations": examination_metadata.get('sources', [])
                        })
                        
                        # Store in global examination history
                        store_examination(examination_name, int(answer) if answer.isdigit() else 1)
                
                exam_count += 1
                
                # Use the EXACT same judge_exam logic from chad.py
                if judge_exam(exam_responses, examination_name if 'examination_name' in locals() else "examination"):
                    print("\n✅ Sufficient examination data collected.")
                    break
                        
            except Exception as e:
                print(f"Error in examination phase: {e}")
                import traceback
                traceback.print_exc()
                break

        # EXACT Final Diagnosis from chad.py
        print("\n🩺 GENERATING DIAGNOSIS")
        print("=" * 40)
        
        diagnosis_result = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
        
        print(f"\n📋 FINAL RESULTS:")
        print("=" * 50)
        print(f"Diagnosis: {diagnosis_result.get('diagnosis', 'Unable to determine')}")
        print(f"Treatment: {diagnosis_result.get('treatment', 'Unable to determine')}")
        print(f"Confidence: {diagnosis_result.get('confidence', 'Unknown')}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chw_case_session_{timestamp}.json"
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
        print("\n✅ CHW WORKFLOW COMPLETE!")

def main():
    runner = SimplePatientRunner()
    
    print("🏥 CHW WORKFLOW PATIENT TESTER")
    print("=" * 50)
    
    # Show cases
    df = runner.show_cases(25)
    
    # Get case selection
    while True:
        choice = input(f"\nEnter case ID (0-{len(df)-1}) or 'more' to see more cases: ").strip()
        if choice == 'more':
            df = runner.show_cases(50)
            continue
        elif choice.isdigit():
            case_id = int(choice)
            if 0 <= case_id < len(df):
                break
            else:
                print(f"Invalid choice. Enter 0-{len(df)-1}")
        else:
            print("Invalid choice. Enter a number or 'more'")
    
    # Load patient case
    patient_info = runner.load_patient_case(case_id)
    if not patient_info:
        print("❌ Failed to load patient case! Make sure PatientBot API is running.")
        return
    
    print(f"\n👤 LOADED PATIENT CASE {case_id}:")
    print(f"Age: {patient_info['age_years']} years ({patient_info['age_category']})")
    print(f"Sex: {patient_info['sex']}")
    print(f"Primary Complaint: {patient_info['primary_complaint']}")
    print(f"Duration: {patient_info['duration']}")
    
    input("\nPress Enter to start CHW workflow...")
    
    # Run CHW workflow
    runner.run_chw_workflow(patient_info)

if __name__ == "__main__":
    main() 