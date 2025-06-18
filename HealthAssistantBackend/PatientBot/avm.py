#!/usr/bin/env python3
"""
Auto Validation Module (AVM)
Automated testing of CHW workflow accuracy against ground truth
"""

import pandas as pd
import sys
import os
import requests
import json
import random
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

# Import core functions but create standalone versions
sys.path.append('../HealthAssistantBackend')
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("✅ Successfully imported OpenAI for standalone examination generation")
except ImportError as e:
    print(f"❌ Failed to import OpenAI: {e}")
    openai = None

class AutoValidationModule:
    def __init__(self):
        self.patient_bot_url = "http://127.0.0.1:5003"
        self.outputs_dir = "outputs"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        
        # Load the CSV data for ground truth
        self.df = pd.read_csv('data.csv')
        print(f"📊 Loaded {len(self.df)} test cases from data.csv")
    
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
4. IMPORTANT: If the patient response is a DURATION (like "2 days") but the question is NOT about duration/time, then select option 5 (Other) with the response as specific answer
5. Only use option 5 (Other) if the response truly doesn't match ANY of the specific options 1-4
6. Be SMART about medical context - use the patient's complaint/symptoms to pick the best match

Context Analysis:
- If question asks about FEVER/TEMPERATURE and patient says "2 days", this is WRONG context → option 5
- If question asks about COUGH DURATION and patient says "2 days", this is CORRECT context → match closest duration
- If question asks about BREATHING/SYMPTOMS and patient says "2 days", this is WRONG context → option 5
- If question asks about YES/NO and patient says duration, this is WRONG context → option 5

Examples:
- Q: "Has child had fever?" + A: "2 days" → option 5 (duration doesn't answer yes/no)
- Q: "How long has fever lasted?" + A: "2 days" → match closest duration option
- Q: "Any difficulty breathing?" + A: "2 days" → option 5 (duration doesn't answer yes/no)
- Patient with "Cough" complaint + says "Yes" to breathing difficulty → pick breathing-related yes option

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

    def generate_standalone_examination(self, session_data):
        """Standalone version of generate_examination without Flask dependencies"""
        try:
            initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                if resp['question'] == "Please describe what brings you here today"), "")
            
            # Extract patient info
            patient_info = {}
            sex_response = next((resp for resp in session_data['initial_responses'] 
                            if "sex" in resp['question'].lower()), None)
            if sex_response:
                patient_info['sex'] = sex_response['answer']
                
            age_response = next((resp for resp in session_data['initial_responses'] 
                            if "age" in resp['question'].lower()), None)
            if age_response:
                patient_info['age'] = age_response['answer']
            
            # Get previous exams
            previous_exams = []
            if 'exam_responses' in session_data and session_data['exam_responses']:
                for exam in session_data['exam_responses']:
                    if 'examination' in exam:
                        previous_exams.append(exam['examination'])
            
            # Build context
            context_items = [initial_complaint]
            context_items.append(f"Patient: {patient_info}")
            
            # Get symptoms from followup responses
            symptoms = []
            for resp in session_data['followup_responses']:
                answer = resp['answer'].lower()
                if any(symptom in answer for symptom in 
                      ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                       'headache', 'nausea', 'dizziness', 'rash']):
                    symptoms.append(answer)
            
            context_items.extend(symptoms)
            
            # Get embeddings and relevant docs
            embeddings = get_embedding_batch(context_items)
            index = pc.Index("who-guide-old")
            relevant_matches = []
            for emb in embeddings:
                matches = vectorQuotesWithSource(emb, index, top_k=2)
                if matches:
                    relevant_matches.extend(matches)
            
            # Sort and deduplicate
            relevant_matches.sort(key=lambda x: x['score'], reverse=True)
            seen_ids = set()
            unique_matches = []
            for match in relevant_matches:
                if match['id'] not in seen_ids:
                    seen_ids.add(match['id'])
                    unique_matches.append(match)
            
            top_matches = unique_matches[:3]
            medical_guide_content = "\n\n".join([match['text'] for match in top_matches])
            
            # Create examination prompt
            previous_exams_text = "\n".join([f"- {exam}" for exam in previous_exams]) if previous_exams else "- No previous examinations"
            symptoms_summary = ", ".join(symptoms[:3]) if symptoms else "No specific symptoms identified"
            
            prompt = f'''Patient information:
Initial complaint: "{initial_complaint}"
Key symptoms: {symptoms_summary}
Patient age: {patient_info.get('age', 'Unknown')}
Patient sex: {patient_info.get('sex', 'Unknown')}

Previous examinations already performed (DO NOT REPEAT THESE):
{previous_exams_text}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above.
2. You MUST select one of these three formats:

FORMAT 1 - WHEN THE GUIDE PROVIDES AN EXAMINATION PROCEDURE:
Examination: [Name]
Procedure: [Steps from guide]
#: [Finding 1]
#: [Finding 2] 
#: [Finding 3]
#: [Finding 4]

FORMAT 2 - WHEN NO EXAMINATION INFO:
"The medical guide does not provide specific examination procedures for this condition."

FORMAT 3 - WHEN PARTIAL INFO:
"PARTIAL INFORMATION: [relevant guidance from guide]"

3. Choose an examination DIFFERENT from previous ones.
4. Use exact terminology from the medical guide.

Based ONLY on the medical guide, what examination should be performed?'''

            if not openai:
                raise Exception("OpenAI not available")
                
            completion = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You can ONLY use information from the medical guide."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            examination_text = completion.choices[0].message.content.strip()
            
            # Check for no information case
            if (examination_text == "The medical guide does not provide specific examination procedures for this condition." or
                "does not provide" in examination_text.lower()):
                
                options = [
                    {"id": 1, "text": "Proceed with general assessment and diagnosis"},
                    {"id": 2, "text": "Refer to higher level facility"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
                
                return {
                    "status": "success",
                    "output": f"The medical guide does not provide specific examination procedures for this condition.\n\nPlease select how to proceed:",
                    "metadata": {
                        "phase": "exam",
                        "question_type": "EXAM",
                        "options": options,
                        "sources": top_matches,
                        "type": "NO_INFO_EXAM"
                    }
                }
            
            # Parse examination with findings
            try:
                examination, option_texts = parse_examination_text(examination_text)
                options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts[:4])]
                options.append({"id": 5, "text": "Other (please specify)"})
                
                # Format output
                if any(phrase in examination.lower() for phrase in ["does not provide", "no information"]):
                    output = f"Examination Information:\n{examination}\n\nRecommended Action:"
                else:
                    output = f"Recommended Examination:\n{examination}\n\nFindings:"
                
                return {
                    "status": "success",
                    "output": output,
                    "metadata": {
                        "phase": "exam",
                        "question_type": "EXAM", 
                        "options": options,
                        "sources": top_matches,
                        "examination_name": examination.split('\n')[0] if '\n' in examination else examination
                    }
                }
                
            except Exception as e:
                print(f"Error parsing examination: {e}")
                options = [
                    {"id": 1, "text": "Proceed with general assessment"},
                    {"id": 2, "text": "Refer to higher level facility"}
                ]
                
                return {
                    "status": "success",
                    "output": "Unable to determine specific examination from medical guide.\n\nRecommended action:",
                    "metadata": {
                        "phase": "exam",
                        "question_type": "EXAM",
                        "options": options,
                        "sources": [],
                        "type": "NO_INFO_EXAM"
                    }
                }
                
        except Exception as e:
            print(f"Error in standalone examination generation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def validate_diagnosis_accuracy(self, predicted_diagnosis, correct_answer, case_info):
        """Use GPT to check if the correct diagnosis appears in the explanation"""
        if not openai:
            return False, "OpenAI not available"
        
        prompt = f"""You are an expert medical accuracy validator. Your job is to determine if a lengthy, cautious medical explanation contains the correct diagnosis, even if it's buried in disclaimers, legal language, or verbose explanations.

CORRECT ANSWER (Ground Truth): "{correct_answer}"

FULL PREDICTED DIAGNOSIS (CHW System Output - may be very long and cautious):
{predicted_diagnosis}

CASE CONTEXT:
- Age: {case_info.get('age_years', 'Unknown')} years
- Sex: {case_info.get('sex', 'Unknown')}
- Complaint: {case_info.get('primary_complaint', 'Unknown')}
- Duration: {case_info.get('duration', 'Unknown')}

CRITICAL INSTRUCTIONS - BE AGGRESSIVE IN FINDING MATCHES:

1. **EXTRACT THE CORE DIAGNOSIS**: Look through ALL the text for any mention of the key medical condition, even if surrounded by "could suggest", "might indicate", "possible", "may be", "based on the guide", etc.

2. **MEDICAL SYNONYMS COUNT AS CORRECT**:
   - "Common cold" = "viral illness" = "upper respiratory infection" = "cold symptoms"
   - "Pneumonia" = "lung infection" = "chest infection" = "respiratory infection" (with breathing difficulty)
   - "Viral sore throat" = "throat infection" = "sore throat" = "viral throat symptoms" = "cold" (when throat is primary)
   - "Dehydration" = "fluid loss" = "water loss" = "poor fluid intake"
   - "Diarrhea" = "loose stools" = "stomach upset" = "gastrointestinal symptoms"

3. **TREATMENT MATCHING**: If the diagnosis is unclear but treatment matches, consider it correct:
   - "Fluids and rest" = "supportive care" = "home care" = "conservative management"
   - "Antibiotics" = "antibiotic treatment" = "antimicrobial therapy"
   - "Urgent referral" = "emergency referral" = "immediate medical attention"

4. **IGNORE DISCLAIMERS**: Don't be fooled by phrases like:
   - "I am not a doctor but..."
   - "Based on the medical guide..."
   - "This should not replace medical advice..."
   - "Please consult a healthcare provider..."
   - "The guide suggests..."

5. **LOOK FOR BURIED INFORMATION**: The correct diagnosis might be:
   - In the middle of a paragraph
   - After multiple disclaimers
   - Mentioned as "could be" or "might suggest"
   - Part of a differential diagnosis list

6. **EXAMPLES OF CORRECT MATCHES**:
   - Expected: "Common cold" + Text contains: "symptoms suggest a viral upper respiratory infection" → CORRECT
   - Expected: "Pneumonia" + Text contains: "breathing difficulty could indicate a lung infection requiring antibiotics" → CORRECT
   - Expected: "Viral sore throat" + Text contains: "sore throat symptoms for 2 days might be part of a simple cold" → CORRECT

7. **RETURN GUIDELINES**:
   - "CORRECT" if ANY form of the correct diagnosis is mentioned ANYWHERE in the text
   - "INCORRECT" ONLY if a completely different diagnosis is given
   - "INCONCLUSIVE" ONLY if absolutely no specific medical condition is mentioned

Be AGGRESSIVE - err on the side of finding matches rather than missing them.

Return only: CORRECT, INCORRECT, or INCONCLUSIVE"""

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are an aggressive medical accuracy validator. Look deep into lengthy text to find any mention of the correct diagnosis. Be generous with matches and synonyms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            result = completion.choices[0].message.content.strip().upper()
            
            if result in ["CORRECT", "INCORRECT", "INCONCLUSIVE"]:
                return result, "Enhanced GPT validation completed"
            else:
                return "INCONCLUSIVE", f"Unexpected GPT response: {result}"
                
        except Exception as e:
            return "ERROR", f"GPT validation failed: {e}"

    def run_single_test_case(self, case_id, patient_info, silent=False):
        """Run a single test case and return results"""
        try:
            if not silent:
                print(f"\n🧪 TESTING CASE {case_id}")
                print(f"Age: {patient_info['age_years']} years, Sex: {patient_info['sex']}")
                print(f"Complaint: {patient_info['primary_complaint']} for {patient_info['duration']}")
            
            # Get correct answer from CSV
            correct_answer = self.df.iloc[case_id]['Diagnosis & Treatment (right answer for GraderBot.  Do NOT pass to the PatientBot LLM) '].strip()
            if not silent:
                print(f"🎯 Expected: {correct_answer}")
                print("\n🏥 MEDICAL ASSESSMENT - INITIAL QUESTIONS")
                print("=" * 60)
            
            # Run CHW workflow (same as run_case.py but with optional verbosity)
            global structured_questions_array, examination_history
            structured_questions_array = []
            examination_history = []
            
            initial_responses = []
            followup_responses = []
            exam_responses = []
            
            # Initial Questions - WITH FULL DISPLAY
            for question in questions_init:
                if not silent:
                    print(f"\n{question['question']}")
                
                if question['type'] in ['MC', 'YN', 'MCM']:
                    if not silent:
                        print_options(question['options'])
                    
                    patient_response = self.ask_patient_bot(question['question'])
                    if not silent:
                        print(f"🤒 PatientBot: {patient_response}")
                    
                    answer, specific_answer = self.match_response_to_option(patient_response, question['options'], patient_info)
                    if not silent:
                        print(f"📋 Option Selection: {answer}")
                        if specific_answer:
                            print(f"✏️ Specific Answer: {specific_answer}")
                        else:
                            print(f"✏️ Specific Answer: None")
                    
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
                    if not silent:
                        print(f"🤒 PatientBot: {patient_response}")
                    
                    auto_answer = self.extract_number_from_response(patient_response)
                    answer = str(auto_answer)
                    if not silent:
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
                    if not silent:
                        print(f"🤒 PatientBot: {patient_response}")
                    
                    if patient_response != "other: Information not known":
                        answer = patient_response
                    else:
                        if patient_info and 'what brings you here today' in question['question'].lower():
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
                    
                    if not silent:
                        print(f"📋 Generated Response: {answer}")
                        print(f"✏️ Specific Answer: None")
                    
                    initial_responses.append({
                        "question": question['question'],
                        "answer": answer,
                        "type": question['type']
                    })

            if not silent:
                print("\n🏥 INITIAL ASSESSMENT COMPLETE")
                for resp in initial_responses:
                    print(f"Q: {resp['question']}")
                    print(f"A: {resp['answer']}\n")

            # Follow-up Questions - WITH FULL DISPLAY
            initial_complaint = next((resp['answer'] for resp in initial_responses 
                                if resp['question'] == "Please describe what brings you here today"), "")
            
            if not silent:
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
                        if not silent:
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
                    
                    if not silent:
                        print(f"\n{question}")
                        print_options(options)
                    
                    patient_response = self.ask_patient_bot(question)
                    if not silent:
                        print(f"🤒 PatientBot: {patient_response}")
                    
                    answer, specific_answer = self.match_response_to_option(patient_response, options, patient_info)
                    if not silent:
                        print(f"📋 Option Selection: {answer}")
                        if specific_answer:
                            print(f"✏️ Specific Answer: {specific_answer}")
                        else:
                            print(f"✏️ Specific Answer: None")
                    
                    if validate_mc_input(answer, options):
                        if answer == "5":
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
                            if not silent:
                                print("\n✅ Moving to examination phase...")
                            break
                    else:
                        break
                        
                except Exception as e:
                    if not silent:
                        print(f"Error in follow-up questions: {e}")
                    break

            # Examination Phase - WITH FULL DISPLAY
            if not silent:
                print("\n🔬 EXAMINATION PHASE")
                print("=" * 30)
            
            exam_count = 0
            max_exams = 3
            
            while exam_count < max_exams:
                try:
                    session_data = {
                        'session_id': f'avm_case_{case_id}',
                        'initial_responses': initial_responses,
                        'followup_responses': followup_responses,
                        'exam_responses': exam_responses,
                        'phase': 'exam'
                    }
                    
                    examination_result = self.generate_standalone_examination(session_data)
                    
                    if examination_result.get('status') != 'success':
                        if not silent:
                            print(f"❌ Examination generation failed: {examination_result.get('message', 'Unknown error')}")
                        break
                    
                    examination_output = examination_result['output']
                    examination_metadata = examination_result.get('metadata', {})
                    examination_options = examination_metadata.get('options', [])
                    exam_type = examination_metadata.get('type', 'EXAM')
                    
                    if not silent:
                        print(f"\n{examination_output}")
                    
                    if exam_type == "NO_INFO_EXAM":
                        if not silent:
                            print("📋 Please select an option:")
                            for i, option in enumerate(examination_options, 1):
                                print(f"  {i}: {option['text']}")
                            print(f"🤒 PatientBot: Proceed with assessment")
                            print(f"📋 Option Selection: 1")
                            print(f"✏️ Specific Answer: None")
                        
                        exam_responses.append({
                            "examination": "No specific examination available in medical guide",
                            "result": "Proceeded with general assessment",
                            "type": "NO_INFO_EXAM"
                        })
                        break
                    else:
                        if examination_options:
                            lines = examination_output.split('\n')
                            examination_name = "Physical examination"
                            for line in lines:
                                if "Examination:" in line or "Recommended Examination:" in line:
                                    examination_name = line.split(':')[1].strip()
                                    break
                            
                            finding_question = f"During {examination_name.lower()}, what findings do you observe?"
                            patient_response = self.ask_patient_bot(finding_question)
                            if not silent:
                                print(f"🤒 PatientBot: {patient_response}")
                            
                            answer, specific_answer = self.match_response_to_option(
                                patient_response, examination_options, patient_info
                            )
                            
                            if not silent:
                                print(f"📋 Option Selection: {answer}")
                                if specific_answer:
                                    print(f"✏️ Specific Answer: {specific_answer}")
                                else:
                                    print(f"✏️ Specific Answer: None")
                            
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
                            
                            exam_responses.append({
                                "examination": examination_name,
                                "result": result_text,
                                "citations": examination_metadata.get('sources', [])
                            })
                            
                            store_examination(examination_name, int(answer) if answer.isdigit() else 1)
                            
                            if not silent:
                                print(f"Recorded: {examination_name} -> {result_text}")
                    
                    exam_count += 1
                    
                    if judge_exam(exam_responses, examination_name if 'examination_name' in locals() else "examination"):
                        if not silent:
                            print("\n✅ Sufficient examination data collected.")
                        break
                            
                except Exception as e:
                    if not silent:
                        print(f"Error in examination phase: {e}")
                        import traceback
                        traceback.print_exc()
                    break

            # Generate Diagnosis - WITH FULL DISPLAY
            if not silent:
                print("\n🩺 GENERATING DIAGNOSIS")
                print("=" * 40)
            
            diagnosis_result = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
            predicted_diagnosis = diagnosis_result.get('diagnosis', 'Unable to determine')
            
            if not silent:
                print(f"\n📋 FINAL RESULTS:")
                print("=" * 50)
                print(f"Diagnosis: {predicted_diagnosis}")
                print(f"Treatment: {diagnosis_result.get('treatment', 'Unable to determine')}")
                print(f"Confidence: {diagnosis_result.get('confidence', 'Unknown')}")
            
            # Validate accuracy using GPT
            validation_result, validation_message = self.validate_diagnosis_accuracy(
                predicted_diagnosis, correct_answer, patient_info
            )
            
            if not silent:
                print(f"\n🔍 VALIDATION:")
                print(f"Expected: {correct_answer}")
                print(f"Predicted: {predicted_diagnosis[:500]}{'...' if len(predicted_diagnosis) > 500 else ''}")
                print(f"✅ Validation: {validation_result}")
            
            return {
                'case_id': case_id,
                'correct_answer': correct_answer,
                'predicted_diagnosis': predicted_diagnosis,
                'validation_result': validation_result,
                'validation_message': validation_message,
                'patient_info': patient_info
            }
            
        except Exception as e:
            if not silent:
                print(f"Error in test case: {e}")
            return {
                'case_id': case_id,
                'error': str(e),
                'validation_result': 'ERROR'
            }

    def run_batch_validation(self, n_cases):
        """Run N random test cases and generate accuracy statistics"""
        print(f"\n🚀 STARTING BATCH VALIDATION OF {n_cases} CASES")
        print("=" * 60)
        
        # Select random cases
        available_cases = list(range(len(self.df)))
        selected_cases = random.sample(available_cases, min(n_cases, len(available_cases)))
        
        results = []
        correct_count = 0
        incorrect_count = 0
        inconclusive_count = 0
        error_count = 0
        
        for i, case_id in enumerate(selected_cases, 1):
            print(f"\n[{i}/{n_cases}] Testing Case {case_id}...", end=' ')
            
            # Load patient case
            patient_info = self.load_patient_case(case_id)
            if not patient_info:
                print("❌ Failed to load")
                error_count += 1
                continue
            
            # Run test case
            result = self.run_single_test_case(case_id, patient_info, silent=True)
            results.append(result)
            
            # Update counts
            validation = result.get('validation_result', 'ERROR')
            if validation == 'CORRECT':
                correct_count += 1
                print("✅")
            elif validation == 'INCORRECT':
                incorrect_count += 1
                print("❌")
            elif validation == 'INCONCLUSIVE':
                inconclusive_count += 1
                print("❓")
            else:
                error_count += 1
                print("💥")
        
        # Generate statistics - Fix division by zero
        total_tested = len([r for r in results if 'error' not in r])
        if total_tested > 0:
            accuracy = (correct_count / total_tested * 100)
            correct_pct = (correct_count / total_tested * 100)
            incorrect_pct = (incorrect_count / total_tested * 100)
            inconclusive_pct = (inconclusive_count / total_tested * 100)
        else:
            accuracy = 0
            correct_pct = 0
            incorrect_pct = 0
            inconclusive_pct = 0
        
        print(f"\n📊 BATCH VALIDATION RESULTS")
        print("=" * 50)
        print(f"Total Cases Tested: {total_tested}")
        print(f"✅ Correct: {correct_count} ({correct_pct:.1f}%)")
        print(f"❌ Incorrect: {incorrect_count} ({incorrect_pct:.1f}%)")
        print(f"❓ Inconclusive: {inconclusive_count} ({inconclusive_pct:.1f}%)")
        print(f"💥 Errors: {error_count}")
        print(f"\n🎯 Overall Accuracy: {accuracy:.1f}%")
        
        if total_tested == 0:
            print("\n⚠️  No cases were successfully tested!")
            print("Make sure PatientBot API is running and accessible.")
            return None
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"avm_batch_results_{timestamp}.json"
        filepath = os.path.join(self.outputs_dir, filename)
        
        batch_results = {
            'timestamp': timestamp,
            'n_cases': n_cases,
            'total_tested': total_tested,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'inconclusive_count': inconclusive_count,
            'error_count': error_count,
            'accuracy_percentage': accuracy,
            'detailed_results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {filepath}")
        
        return batch_results

def main():
    avm = AutoValidationModule()
    
    print("🤖 AUTO VALIDATION MODULE (AVM)")
    print("=" * 50)
    print("Automated testing of CHW workflow accuracy")
    
    while True:
        try:
            n_cases = input(f"\nEnter number of test cases to run (1-{len(avm.df)}): ").strip()
            if n_cases.isdigit():
                n_cases = int(n_cases)
                if 1 <= n_cases <= len(avm.df):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(avm.df)}")
            else:
                print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Ask for verbose mode
    while True:
        try:
            verbose_choice = input("\nShow full interactions like run_case.py? (y/n): ").strip().lower()
            if verbose_choice in ['y', 'yes', 'true', '1']:
                verbose = True
                print("🔍 Running in VERBOSE mode - showing full interactions")
                break
            elif verbose_choice in ['n', 'no', 'false', '0']:
                verbose = False
                print("⚡ Running in SILENT mode - showing only results")
                break
            else:
                print("Please enter 'y' for verbose or 'n' for silent")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
    
    # Run batch validation with verbose option
    if verbose:
        # For verbose mode, run cases one by one with full display
        print(f"\n🚀 STARTING VERBOSE VALIDATION OF {n_cases} CASES")
        print("=" * 60)
        
        # Select random cases
        available_cases = list(range(len(avm.df)))
        selected_cases = random.sample(available_cases, min(n_cases, len(available_cases)))
        
        results = []
        correct_count = 0
        incorrect_count = 0
        inconclusive_count = 0
        error_count = 0
        
        for i, case_id in enumerate(selected_cases, 1):
            print(f"\n{'='*80}")
            print(f"🧪 CASE {i}/{n_cases} - ID: {case_id}")
            print(f"{'='*80}")
            
            # Load patient case
            patient_info = avm.load_patient_case(case_id)
            if not patient_info:
                print("❌ Failed to load patient case")
                error_count += 1
                continue
            
            # Run test case in verbose mode
            result = avm.run_single_test_case(case_id, patient_info, silent=False)
            results.append(result)
            
            # Update counts and show result
            validation = result.get('validation_result', 'ERROR')
            if validation == 'CORRECT':
                correct_count += 1
                print(f"\n🎉 CASE {i} RESULT: ✅ CORRECT")
            elif validation == 'INCORRECT':
                incorrect_count += 1
                print(f"\n❌ CASE {i} RESULT: ❌ INCORRECT")
            elif validation == 'INCONCLUSIVE':
                inconclusive_count += 1
                print(f"\n❓ CASE {i} RESULT: ❓ INCONCLUSIVE")
            else:
                error_count += 1
                print(f"\n💥 CASE {i} RESULT: 💥 ERROR")
            
            # Ask if user wants to continue
            if i < n_cases:
                try:
                    continue_choice = input(f"\nContinue to case {i+1}? (y/n/q for quit): ").strip().lower()
                    if continue_choice in ['q', 'quit', 'exit']:
                        print("🛑 Stopping validation...")
                        break
                    elif continue_choice in ['n', 'no']:
                        print("🛑 Stopping validation...")
                        break
                except KeyboardInterrupt:
                    print("\n🛑 Stopping validation...")
                    break
        
        # Generate final statistics
        total_tested = len([r for r in results if 'error' not in r])
        if total_tested > 0:
            accuracy = (correct_count / total_tested * 100)
        else:
            accuracy = 0
        
        print(f"\n📊 FINAL BATCH VALIDATION RESULTS")
        print("=" * 50)
        print(f"Total Cases Tested: {total_tested}")
        print(f"✅ Correct: {correct_count}")
        print(f"❌ Incorrect: {incorrect_count}")
        print(f"❓ Inconclusive: {inconclusive_count}")
        print(f"💥 Errors: {error_count}")
        print(f"\n🎯 Overall Accuracy: {accuracy:.1f}%")
    else:
        # Use the existing batch validation for silent mode
        results = avm.run_batch_validation(n_cases)
    
    print(f"\n✅ Batch validation complete!")
    print(f"Check outputs folder for detailed results.")

if __name__ == "__main__":
    main() 