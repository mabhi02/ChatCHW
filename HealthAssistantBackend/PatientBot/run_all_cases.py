#!/usr/bin/env python3
"""
Run specified number of patient cases through CHW workflow and store results with grading
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

def grade_aspect(case_results, rubric_row, column_name):
    """Grade a specific aspect of the case against the rubric using GPT-4"""
    
    # Special handling for True diagnosis and Presents with
    if column_name in ['True diagnosis', 'Presents with']:
        content_to_grade = case_results.get('diagnosis', {}).get('diagnosis', '')
    else:
        content_to_grade = case_results
    
    prompt = f"""You are a medical grading system. Compare the AI's responses with the rubric criteria and determine a score.

GRADING RULES:
1. Return "1" if the AI's responses fully match the rubric criteria
2. Return "0.5" if the AI's responses partially match the criteria
3. Return "0" if they do not match 
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
        return 0  # Default to 0 on error

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
    
    # Create graded directory if it doesn't exist
    graded_dir = "graded"
    if not os.path.exists(graded_dir):
        os.makedirs(graded_dir)
        print(f"[DEBUG] Created graded directory: {graded_dir}")
    
    # Load mapped data and rubric
    print("[DEBUG] Loading mapped_data.csv...")
    try:
        df = pd.read_csv('mapped_data.csv')
        print(f"[DEBUG] Successfully loaded mapped_data.csv with {len(df)} rows")
    except Exception as e:
        print(f"[DEBUG] Error loading mapped_data.csv: {e}")
        return
        
    print("[DEBUG] Loading grading rubric...")
    try:
        rubric_df = pd.read_csv('Grading rubric WHO 2012 ChatGPT (1).xlsx - Rubric  (1).csv')
        print(f"[DEBUG] Successfully loaded rubric with {len(rubric_df)} rows")
    except Exception as e:
        print(f"[DEBUG] Error loading rubric: {e}")
        return
    
    # Get number of cases to run
    num_cases = get_num_cases()
    print(f"[DEBUG] Will process {num_cases} cases")
    
    # Initialize results dictionary
    all_results = {}
    
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
            
            # Get corresponding rubric row
            rubric_row_idx = int(row['Mapped_Rubric_Row']) - 1  # Convert to 0-based index
            rubric_row = rubric_df.iloc[rubric_row_idx]
            
            # Grade each aspect
            grades = {}
            for column in rubric_df.columns:
                grade = grade_aspect(case_results, rubric_row, column)
                grades[f"grading_{column}"] = grade
                print(f"[DEBUG] Graded {column}: {grade}")
            
            # Add results to case data
            case_results.update(grades)
            
            # Store in all_results
            all_results[f"Case_{case_id}"] = case_results
            print(f"[DEBUG] Stored results for case {case_id}")
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_cases_results_{num_cases}cases_{timestamp}.json"
    filepath = os.path.join(graded_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    print("\n✅ ALL CASES COMPLETE!")

if __name__ == "__main__":
    main() 