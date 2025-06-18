#!/usr/bin/env python3
"""
Convert graded JSON files to CSV format matching ChatCHW_Table.csv structure
"""

import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv

def extract_initial_responses(case_data):
    """Extract initial responses from case data"""
    responses = []
    if 'initial_responses' in case_data:
        for resp in case_data['initial_responses']:
            question = resp.get('question', '')
            answer = resp.get('answer', '')
            if 'what brings you here today' in question.lower():
                responses.append(['Question', f'[{question}]', str(answer), 'NA', '[100%]'])
            elif 'age' in question.lower():
                responses.append(['Question', f'[{question}]', str(answer), 'Age', '[100%]'])
            elif 'sex' in question.lower():
                responses.append(['Question', f'[{question}]', str(answer), 'sex', '[100%]'])
    return responses

def extract_followup_responses(case_data):
    """Extract followup responses from case data"""
    responses = []
    if 'followup_responses' in case_data:
        for resp in case_data['followup_responses']:
            question = resp.get('question', '')
            answer = resp.get('answer', '')
            responses.append(['Questions', question, str(answer), '—', '95%'])
    return responses

def extract_exam_responses(case_data):
    """Extract examination responses from case data"""
    responses = []
    if 'exam_responses' in case_data:
        responses.append(['Exams', '', '', '', ''])
        for resp in case_data['exam_responses']:
            exam = resp.get('examination', '')
            result = resp.get('result', '')
            responses.append(['Exam', exam, str(result), '—', '95%'])
    return responses

def extract_diagnosis_and_treatment(case_data):
    """Extract diagnosis and treatment information"""
    responses = []
    if 'diagnosis' in case_data:
        diagnosis = case_data['diagnosis']
        if isinstance(diagnosis, dict):
            if 'diagnosis' in diagnosis:
                responses.append(['Diagnosis', '', diagnosis['diagnosis'], '—', '95%'])
            if 'treatment' in diagnosis:
                treatment_text = diagnosis['treatment']
                # Split treatment text into sections
                for line in treatment_text.split('\n'):
                    if line.strip():
                        responses.append(['Treatment', '', line.strip(), '—', '100%'])
        elif isinstance(diagnosis, str):
            responses.append(['Diagnosis', '', diagnosis, '—', '95%'])
    return responses

def convert_json_to_csv(json_path, output_path):
    """Convert a JSON file to CSV format matching ChatCHW_Table.csv structure"""
    
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get case data
    case_data = data.get('case_data', data)  # Handle both new and old format
    
    # Initialize rows with headers
    rows = [['Section', 'ChatCHW asks', 'Reply', 'GraderBot rubric', 'Similarity']]
    
    # Add case ID (using timestamp)
    case_id = case_data.get('timestamp', Path(json_path).stem)
    rows.append(['New patient', '', case_id, 'NA', '—'])
    
    # Add all sections
    rows.extend(extract_initial_responses(case_data))
    rows.extend(extract_followup_responses(case_data))
    rows.extend(extract_exam_responses(case_data))
    rows.extend(extract_diagnosis_and_treatment(case_data))
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('graded/csv_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all testing1 folders in graded directory
    graded_dir = Path('graded')
    for folder in graded_dir.glob('testing1*'):
        if folder.is_dir():
            print(f"Processing folder: {folder}")
            
            # Create corresponding output folder
            folder_output_dir = output_dir / folder.name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each JSON file in the folder
            for json_file in folder.glob('*.json'):
                print(f"Converting file: {json_file}")
                output_file = folder_output_dir / f"{json_file.stem}.csv"
                try:
                    convert_json_to_csv(json_file, output_file)
                    print(f"✅ Created CSV file: {output_file}")
                except Exception as e:
                    print(f"❌ Error processing {json_file}: {e}")

if __name__ == "__main__":
    main() 