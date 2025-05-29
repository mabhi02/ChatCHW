#!/usr/bin/env python3
"""
Run specified number of patient cases through CHW workflow and store results with correct answers
"""

import pandas as pd
import sys
import os
import json
from datetime import datetime
from tqdm import tqdm
import openai
from avm import SimplePatientRunner

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

def grade_diagnosis(ai_diagnosis, correct_answer):
    """Grade the AI's diagnosis against the correct answer using OpenAI"""
    
    print(f"\n[DEBUG] Grading diagnosis:")
    print(f"[DEBUG] AI Diagnosis: {ai_diagnosis}")
    print(f"[DEBUG] Correct Answer: {correct_answer}")
    
    prompt = f"""You are a medical grading system. Compare the AI's diagnosis with the correct answer and determine if they match.

GRADING RULES:
1. Return EXACTLY "1" if the AI's diagnosis captures the same medical condition(s) as the correct answer
2. Return EXACTLY "0" if they do not match
3. Ignore minor differences in wording/phrasing
4. Focus on the core medical condition(s) being identified
5. Treatment plans don't need to match exactly, focus on the diagnosis
6. If the correct answer mentions multiple conditions, the AI should identify ALL of them to get a 1
7. If either answer is unclear or non-specific, return 0

AI Diagnosis:
{ai_diagnosis}

Correct Answer:
{correct_answer}

Return ONLY a single digit: 1 for match, 0 for no match.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a deterministic medical grading system that can ONLY output 0 or 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        
        grade = response.choices[0].message.content.strip()
        print(f"[DEBUG] Grade received: {grade}")
        return int(grade)
    except Exception as e:
        print(f"\n❌ [DEBUG] Grading error: {e}")
        return 0  # Default to 0 on error

def create_grading_array():
    """Create array of grading rubric cases"""
    print("\n[DEBUG] Creating grading array from rubric...")
    
    try:
        rubric_df = pd.read_csv('Grading rubric WHO 2012 ChatGPT (1).xlsx - Rubric  (1).csv')
        print(f"[DEBUG] Successfully loaded rubric CSV with {len(rubric_df)} rows")
    except Exception as e:
        print(f"[DEBUG] Error loading rubric CSV: {e}")
        return [], None
    
    grading_arr = []
    
    # Clean column names by stripping whitespace
    rubric_df.columns = rubric_df.columns.str.strip()
    print(f"[DEBUG] Cleaned column names: {list(rubric_df.columns)}")
    
    for idx, row in rubric_df.iterrows():
        try:
            # Combine first two columns into a single string
            case_str = f"{row['True diagnosis']} - {row['Presents with']}"
            grading_arr.append(case_str)
            print(f"[DEBUG] Added case {idx}: {case_str[:100]}...")  # Print first 100 chars
        except Exception as e:
            print(f"[DEBUG] Error processing row {idx}: {e}")
    
    print(f"[DEBUG] Created grading array with {len(grading_arr)} cases")
    return grading_arr, rubric_df

def find_matching_rubric_case(patient_case, grading_arr):
    """Find the most similar case in the grading rubric using GPT"""
    
    print("\n[DEBUG] Finding matching rubric case...")
    print(f"[DEBUG] Patient case details:")
    print(f"- Complaint: {patient_case.get('Complaint', 'N/A')}")
    print(f"- Duration: {patient_case.get('Duration', 'N/A')}")
    print(f"- CHW Questions: {patient_case.get('CHW Questions', 'N/A')}")
    
    # Concatenate relevant columns from data.csv case
    case_str = f"{patient_case['Complaint']} - Duration: {patient_case['Duration']} - {patient_case['CHW Questions']}"
    print(f"[DEBUG] Concatenated case string: {case_str}")
    
    prompt = f"""Given a patient case and a list of standard cases from a grading rubric, find the index (0-based) of the most similar case.
Return ONLY the index number, no other text.

Patient case:
{case_str}

Standard cases:
{json.dumps(grading_arr, indent=2)}

Return ONLY the index number (0-based) of the most similar case."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a deterministic medical case matcher that can ONLY output a single number."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2
        )
        
        index = int(response.choices[0].message.content.strip())
        print(f"[DEBUG] Matched to case index: {index}")
        print(f"[DEBUG] Matched case: {grading_arr[index]}")
        return index
    except Exception as e:
        print(f"\n❌ [DEBUG] Case matching error: {e}")
        return 0  # Default to first case on error

def get_num_cases():
    """Get number of cases to run from user input"""
    while True:
        try:
            num = input("Enter number of cases to run (1-1000): ").strip()
            num = int(num)
            if 1 <= num <= 1000:
                return num
            print("Please enter a number between 1 and 1000")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("\n[DEBUG] Starting main execution...")
    
    # Create outputs directory if it doesn't exist
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
        print(f"[DEBUG] Created outputs directory: {outputs_dir}")
    
    # Load all cases and create grading array
    print("[DEBUG] Loading data.csv...")
    try:
        df = pd.read_csv('data.csv')
        print(f"[DEBUG] Successfully loaded data.csv with {len(df)} rows")
    except Exception as e:
        print(f"[DEBUG] Error loading data.csv: {e}")
        return
    
    grading_arr, rubric_df = create_grading_array()
    if not grading_arr or rubric_df is None:
        print("[DEBUG] Failed to create grading array. Exiting...")
        return
    
    # Get number of cases to run
    num_cases = get_num_cases()
    print(f"[DEBUG] Will process {num_cases} cases")
    
    # Initialize results dictionary
    all_results = {}
    total_correct = 0
    
    # Initialize runner
    print("[DEBUG] Initializing SimplePatientRunner...")
    runner = SimplePatientRunner()
    
    print(f"\n🏥 Running {num_cases} cases...")
    
    # Run through specified number of cases
    for idx, row in tqdm(df.iloc[:num_cases].iterrows(), total=num_cases):
        case_id = idx + 1  # Case IDs are 1-indexed
        print(f"\n[DEBUG] Processing case {case_id}...")
        
        # Load patient case
        patient_info = runner.load_patient_case(case_id)
        if not patient_info:
            print(f"\n❌ [DEBUG] Failed to load case {case_id}. Skipping...")
            continue
        
        print(f"[DEBUG] Successfully loaded patient info for case {case_id}")
        
        # Run the case and get results
        case_results = runner.run_chw_workflow(patient_info)
        if case_results:
            print(f"[DEBUG] Successfully ran CHW workflow for case {case_id}")
            
            # Add correct answer from data.csv
            correct_answer = row.iloc[-1]
            case_results['correct_answer'] = correct_answer
            print(f"[DEBUG] Added correct answer: {correct_answer}")
            
            # Find matching rubric case
            rubric_index = find_matching_rubric_case(row, grading_arr)
            case_results['rubric_case'] = grading_arr[rubric_index]
            print(f"[DEBUG] Added rubric case match: {grading_arr[rubric_index][:100]}...")
            
            # Add the entire matching row from the rubric
            matching_row = rubric_df.iloc[rubric_index].to_dict()
            case_results['rubric_details'] = matching_row
            print("[DEBUG] Added complete rubric details")
            
            # Grade the diagnosis
            ai_diagnosis = case_results['diagnosis'].get('diagnosis', '')
            grade = grade_diagnosis(ai_diagnosis, correct_answer)
            case_results['grade'] = grade
            total_correct += grade
            print(f"[DEBUG] Graded diagnosis: {grade}")
            
            # Store in all_results
            all_results[f"Case_{case_id}"] = case_results
            print(f"[DEBUG] Stored results for case {case_id}")
    
    # Calculate accuracy
    accuracy = (total_correct / len(all_results)) * 100 if all_results else 0
    print(f"\n[DEBUG] Final accuracy: {accuracy:.2f}%")
    
    # Add summary statistics
    summary = {
        "total_cases": len(all_results),
        "correct_diagnoses": total_correct,
        "accuracy": accuracy
    }
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_cases_results_{num_cases}cases_{timestamp}.json"
    filepath = os.path.join(outputs_dir, filename)
    
    final_results = {
        "summary": summary,
        "cases": all_results
    }
    
    print(f"[DEBUG] Saving results to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Results for {len(all_results)} cases saved to {filepath}")
    print(f"📊 Accuracy: {accuracy:.2f}% ({total_correct} correct out of {len(all_results)} cases)")

if __name__ == "__main__":
    main() 