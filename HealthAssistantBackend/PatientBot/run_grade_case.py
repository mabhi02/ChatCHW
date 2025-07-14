#!/usr/bin/env python3
"""
Patient Case Runner with Grading
Runs a single case through CHW workflow and grades the result using the rubric and GPT-4.
"""

import pandas as pd
import sys
import os
import requests
import json
from datetime import datetime
import csv

# Add parent directory to path to import ChatCHW modules
sys.path.append('..')

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
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("✅ Successfully imported OpenAI for standalone examination generation")
except ImportError as e:
    print(f"❌ Failed to import OpenAI: {e}")
    openai = None

class SimplePatientRunner:
    def __init__(self):
        self.patient_bot_url = "http://localhost:5003"
        self.outputs_dir = "outputs"
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
    def show_cases(self, limit=25):
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
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/load", json={"case_id": case_id})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['patient']
            return None
        except Exception as e:
            print(f"❌ Error loading patient: {e}")
            print("Make sure PatientBot API is running: python API.py")
            return None
    def ask_patient_bot(self, question, context=None):
        try:
            payload = {"question": question}
            if context is not None:
                payload["context"] = context
            response = requests.post(f"{self.patient_bot_url}/api/patient/ask", json=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['response']
            return "No"
        except Exception as e:
            return "No"
    def match_response_to_option(self, response, options, patient_info=None):
        if response == "other: Information not known":
            for option in options:
                if option['id'] == 5 or 'other' in option['text'].lower():
                    return str(option['id']), "No information found"
            return str(len(options)), "No information found"
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
            import re
            match = re.search(r'\d+', ai_choice)
            if match:
                choice_num = int(match.group())
                if 1 <= choice_num <= len(options):
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
        for option in options:
            if option['id'] == 5:
                return "5", "No information found"
        return "1", None
    def extract_number_from_response(self, response):
        if response == "other: Information not known":
            return 1
        import re
        numbers = re.findall(r'\d+', response)
        return int(numbers[0]) if numbers else 1
    def run_chw_workflow(self, patient_info=None):
        global structured_questions_array, examination_history
        structured_questions_array = []
        examination_history = []
        initial_responses = []
        followup_responses = []
        exam_responses = []
        previous_qa = []
        print("\n🏥 MEDICAL ASSESSMENT - INITIAL QUESTIONS")
        print("=" * 60)
        for question in questions_init:
            print(f"\n{question['question']}")
            if question['type'] in ['MC', 'YN', 'MCM']:
                print_options(question['options'])
                patient_response = self.ask_patient_bot(question['question'], context=previous_qa)
                print(f"🤒 PatientBot: {patient_response}")
                answer, specific_answer = self.match_response_to_option(patient_response, question['options'], patient_info)
                print(f"📋 Option Selection: {answer}")
                if specific_answer:
                    print(f"✏️ Specific Answer: {specific_answer}")
                else:
                    print(f"✏️ Specific Answer: None")
                previous_qa.append({"question": question['question'], "answer": patient_response})
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
                patient_response = self.ask_patient_bot(question['question'], context=previous_qa)
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
                patient_response = self.ask_patient_bot(question['question'], context=previous_qa)
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
        # --- Follow-up Questions ---
        initial_complaint = next((resp['answer'] for resp in initial_responses if resp['question'] == "Please describe what brings you here today"), "")
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
                matrix_output = process_with_matrix(question, followup_responses)
                print(f"\n{question}")
                print_options(options)
                patient_response = self.ask_patient_bot(question, context=previous_qa)
                print(f"🤒 PatientBot: {patient_response}")
                answer, specific_answer = self.match_response_to_option(patient_response, options, patient_info)
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
                        print("\n✅ Moving to examination phase...")
                        break
                else:
                    break
            except Exception as e:
                print(f"Error in follow-up questions: {e}")
                break
        # --- Examination Phase ---
        print("\n🔬 EXAMINATION PHASE")
        print("=" * 30)
        exam_count = 0
        max_exams = 3
        while exam_count < max_exams:
            try:
                session_data = {
                    'session_id': f'run_case_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    'initial_responses': initial_responses,
                    'followup_responses': followup_responses,
                    'exam_responses': exam_responses,
                    'phase': 'exam'
                }
                examination_result = self.generate_standalone_examination(session_data)
                if examination_result.get('status') != 'success':
                    print(f"❌ Examination generation failed: {examination_result.get('message', 'Unknown error')}")
                    break
                examination_output = examination_result['output']
                examination_metadata = examination_result.get('metadata', {})
                examination_options = examination_metadata.get('options', [])
                print(f"\n{examination_output}")
                exam_type = examination_metadata.get('type', 'EXAM')
                if exam_type == "NO_INFO_EXAM":
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
                        patient_response = self.ask_patient_bot(finding_question, context=previous_qa)
                        print(f"🤒 PatientBot: {patient_response}")
                        answer, specific_answer = self.match_response_to_option(
                            patient_response, examination_options, patient_info
                        )
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
                exam_count += 1
                if judge_exam(exam_responses, examination_name if 'examination_name' in locals() else "examination"):
                    print("\n✅ Sufficient examination data collected.")
                    break
            except Exception as e:
                print(f"Error in examination phase: {e}")
                import traceback
                traceback.print_exc()
                break
        # --- Final Diagnosis ---
        print("\n🩺 GENERATING DIAGNOSIS")
        print("=" * 40)
        diagnosis_result = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
        print(f"\n📋 FINAL RESULTS:")
        print("=" * 50)
        print(f"Diagnosis: {diagnosis_result.get('diagnosis', 'Unable to determine')}")
        print(f"Treatment: {diagnosis_result.get('treatment', 'Unable to determine')}")
        print(f"Confidence: {diagnosis_result.get('confidence', 'Unknown')}")
        return {
            'initial_responses': initial_responses,
            'followup_responses': followup_responses,
            'exam_responses': exam_responses,
            'diagnosis': diagnosis_result,
            'structured_questions': structured_questions_array,
            'examination_history': examination_history
        }

# --- Grading Functionality ---
def grade_aspect(case_results, rubric_row, column_name):
    import openai
    if column_name in ['True diagnosis', 'Presents with']:
        content_to_grade = case_results.get('diagnosis', {}).get('diagnosis', '')
    else:
        content_to_grade = case_results
    prompt = f"""You are a medical grading system. Compare the AI's responses with the rubric criteria and determine a score.

GRADING RULES:
1. Return \"1\" if the AI's responses fully match the rubric criteria
2. Return \"0.5\" if the AI's responses partially match the criteria
3. Return \"0\" if they do not match 
4. Focus on the core medical concepts being evaluated
5. Consider both completeness and correctness

Rubric Criteria for {column_name}:
{rubric_row[column_name]}

AI's Response:
{json.dumps(content_to_grade, indent=2)}

Return ONLY a single number: 1 for full match, 0.5 for partial match, 0 for no match.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a deterministic medical grading system that can ONLY output 0, 0.5, or 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        grade = float(response.choices[0].message.content.strip())
        return grade
    except Exception as e:
        print(f"\n❌ [DEBUG] Grading error for {column_name}: {e}")
        return 0

def format_case_to_csv_rows(case_results, rubric_row, rubric_df):
    """
    Use OpenAI to generate rows for the CSV in the format:
    Section | ChatCHW asks | Reply | GraderBot rubric | Similarity
    Returns a list of lists (rows).
    """
    import openai
    prompt = f"""
Given the following case results and rubric row, generate a table with columns:
Section | ChatCHW asks | Reply | GraderBot rubric | Similarity

- Section: e.g. 'Question', 'Exam', 'Diagnosis', 'Treatment', etc.
- ChatCHW asks: The question or prompt asked by the system.
- Reply: The system's reply.
- GraderBot rubric: The corresponding rubric item or NA.
- Similarity: A percentage or score (e.g. 100%, 95%, 0) based on how well the reply matches the rubric, using the grading logic.

Case results (JSON):
{json.dumps(case_results, indent=2)}

Rubric row (JSON):
{json.dumps(rubric_row.to_dict(), indent=2)}

Return the table as CSV rows (no header, one row per line, comma-separated, escape commas in text with quotes if needed).
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a medical grading assistant that formats case results as a CSV table for human review."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1200
        )
        csv_text = response.choices[0].message.content.strip()
        # Parse the CSV text into rows
        import io
        reader = csv.reader(io.StringIO(csv_text))
        rows = [row for row in reader if row]
        return rows
    except Exception as e:
        print(f"\n❌ [DEBUG] CSV formatting error: {e}")
        return []

def main():
    runner = SimplePatientRunner()
    print("🏥 CHW WORKFLOW PATIENT TESTER WITH GRADING")
    print("=" * 50)
    df = runner.show_cases(25)
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
    case_results = runner.run_chw_workflow(patient_info)
    rubric_path = 'Grading rubric WHO 2012 ChatGPT (1).xlsx - Rubric  (1).csv'
    try:
        rubric_df = pd.read_csv(rubric_path)
    except Exception as e:
        print(f"❌ Could not load rubric: {e}")
        return
    # --- Rubric row selection automation ---
    mapped_data_path = 'mapped_data.csv'
    mapped_row = None
    mapped_rubric_row = None
    try:
        mapped_df = pd.read_csv(mapped_data_path)
        if case_id < len(mapped_df):
            mapped_row = mapped_df.iloc[case_id]
            mapped_rubric_row = mapped_row.get('Mapped_Rubric_Row', None)
    except Exception as e:
        print(f"❌ Could not load mapped_data.csv: {e}")
    rubric_row_idx = None
    if mapped_rubric_row is not None and not pd.isna(mapped_rubric_row):
        try:
            rubric_row_idx = int(mapped_rubric_row) - 1
            print(f"[INFO] Auto-selected rubric row {rubric_row_idx+1} for this case based on mapped_data.csv.")
        except Exception as e:
            print(f"[WARN] Could not parse mapped rubric row: {e}")
    if rubric_row_idx is None or rubric_row_idx < 0 or rubric_row_idx >= len(rubric_df):
        print("[WARN] Could not auto-select rubric row. Please enter manually.")
        rubric_row_idx = int(input(f"Enter rubric row number for this case (1-{len(rubric_df)}): ")) - 1
    rubric_row = rubric_df.iloc[rubric_row_idx]
    grades = {}
    print("\n--- GRADING RESULTS ---")
    for column in rubric_df.columns:
        grade = grade_aspect(case_results, rubric_row, column)
        grades[f"grading_{column}"] = grade
        print(f"{column}: {grade}")
    # --- Output folder logic ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    output_dir = os.path.join("7/14-outputs", f"Case{case_id}-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    # 1. Save JSON
    json_path = os.path.join(output_dir, f"Case{case_id}-{timestamp}-output.json")
    with open(json_path, 'w') as f:
        json.dump({"case_id": case_id, "case_results": case_results, "grades": grades}, f, indent=2)
    print(f"\n💾 Graded results saved to: {json_path}")
    # 2. Save CSV
    csv_rows = format_case_to_csv_rows(case_results, rubric_row, rubric_df)
    csv_path = os.path.join(output_dir, f"Case{case_id}-{timestamp}-graded.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "ChatCHW asks", "Reply", "GraderBot rubric", "Similarity"])
        for row in csv_rows:
            writer.writerow(row)
    print(f"💾 Graded CSV saved to: {csv_path}")
    print("\n✅ CASE COMPLETE!")

if __name__ == "__main__":
    main() 