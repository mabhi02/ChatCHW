#!/usr/bin/env python3
"""
Simple access interface for running patient cases
"""

from run_case import SimplePatientRunner
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv

def getRun(row_number):
    """
    Run a specific case by row number and save results
    Args:
        row_number: The row number from data.csv to run
        
    Returns:
        Dictionary containing:
        - patient_info: The loaded patient information
        - runner: The SimplePatientRunner instance
        - case_data: The full case data from data.csv
        - results: Results from running the case
        - output_file: Path to the saved results file
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize the runner
    runner = SimplePatientRunner()
    
    # Load the case data
    df = pd.read_csv('data.csv')
    if row_number < 0 or row_number >= len(df):
        raise ValueError(f"Row number {row_number} is out of range. Must be between 0 and {len(df)-1}")
    
    case_data = df.iloc[row_number].to_dict()
    
    # Load the patient case
    patient_info = runner.load_patient_case(row_number)
    if not patient_info:
        raise RuntimeError("Failed to load patient case. Make sure PatientBot API is running (python API.py)")
    
    # Run the CHW workflow
    print(f"\nRunning case {row_number}...")
    results = runner.run_chw_workflow(patient_info)
    
    # Ensure we have valid results
    if not results:
        results = {
            'initial_responses': [],
            'followup_responses': [],
            'exam_responses': [],
            'diagnosis': {},
            'structured_questions': [],
            'examination_history': []
        }
    
    # Format results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "timestamp": timestamp,
        "case_number": row_number,
        "initial_responses": results.get('initial_responses', []),
        "followup_responses": results.get('followup_responses', []),
        "exam_responses": results.get('exam_responses', []),
        "diagnosis": results.get('diagnosis', {}),
        "structured_questions": results.get('structured_questions', []),
        "examination_history": results.get('examination_history', [])
    }
    
    # Save results to outputs folder
    os.makedirs("outputs", exist_ok=True)
    output_file = os.path.join("outputs", f"chw_case_session_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ CHW WORKFLOW COMPLETE!")
    
    return {
        "patient_info": patient_info,
        "runner": runner,
        "case_data": case_data,
        "results": results,
        "output_file": output_file
    }

def listCases(limit=25):
    """
    Show available cases with their row numbers
    Args:
        limit: Maximum number of cases to show
    """
    runner = SimplePatientRunner()
    runner.show_cases(limit)

def test_case_1():
    """
    Test function to run case #1 from data.csv
    """
    print("\nRunning test case #1...")
    try:
        result = getRun(1)
        print("\nTest successful!")
        print(f"Results saved to: {result['output_file']}")
        return result
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Run test case #1
    test_case_1()
    
    print("\nExample usage:")
    print("from access_run import getRun, listCases")
    print("listCases()  # Show available cases")
    print("result = getRun(0)  # Run case #0")
    print("runner = result['runner']  # Get the runner instance")
    print("patient_info = result['patient_info']  # Get patient info")
    print("case_data = result['case_data']  # Get full case data")
    print("results = result['results']  # Get workflow results")
    print("output_file = result['output_file']  # Get path to saved results") 