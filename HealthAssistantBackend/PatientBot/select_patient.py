#!/usr/bin/env python3
"""
Patient Case Selection Tool
Browse and select specific patient cases from data.csv for testing
"""

import pandas as pd
import requests
import json

def show_csv_data(limit=25):
    """Show CSV data with proper indexing"""
    df = pd.read_csv('data.csv')
    
    print("📋 PATIENT CASES FROM CSV:")
    print("=" * 100)
    print(f"{'ID':<3} {'Row':<4} {'Age':<3} {'Sex':<6} {'Complaint':<20} {'Duration':<15} {'CHW Questions':<30}")
    print("-" * 100)
    
    for i in range(min(limit, len(df))):
        row = df.iloc[i]
        chw_preview = str(row['CHW Questions'])[:27] + "..." if len(str(row['CHW Questions'])) > 30 else str(row['CHW Questions'])
        print(f"{i:<3} {i+2:<4} {row['age (years)']:<3} {row['Sex']:<6} {str(row['Complaint']):<20} {str(row['Duration']):<15} {chw_preview:<30}")
    
    print(f"\nShowing first {min(limit, len(df))} cases out of {len(df)} total cases")
    return df

def show_detailed_case(df, case_id):
    """Show detailed information for a specific case"""
    if case_id >= len(df):
        print(f"Invalid case ID. Max case ID is {len(df)-1}")
        return None
        
    row = df.iloc[case_id]
    
    print(f"\n🔍 DETAILED VIEW - Case ID {case_id} (CSV Row {case_id + 2}):")
    print("=" * 80)
    print(f"Age Category: {row['Age']}")
    print(f"Age (years): {row['age (years)']}")
    print(f"Sex: {row['Sex']}")
    print(f"Complaint: {row['Complaint']}")
    print(f"Duration: {row['Duration']}")
    print(f"\nCHW Questions:")
    print(f"  {row['CHW Questions']}")
    print(f"\nExam Findings:")
    print(f"  {row['Exam Findings']}")
    
    return row

def test_patientbot_with_case(case_id):
    """Test PatientBot API with specific case"""
    try:
        # Load the case
        response = requests.post("http://localhost:5003/api/patient/load", 
                               json={"case_id": case_id})
        if response.status_code != 200:
            print("❌ PatientBot API not running. Start it with: python API.py")
            return
            
        print(f"\n🤖 TESTING PATIENTBOT WITH CASE {case_id}:")
        print("=" * 60)
        
        # Test some questions
        test_questions = [
            "What is your age?",
            "Are you male or female?", 
            "What is wrong with you?",
            "How long have you had this problem?",
            "Do you have a cough?",
            "Do you have fever?",
            "What is your temperature?",
            "Do you have difficulty breathing?",
            "Do you have a runny nose?"
        ]
        
        for question in test_questions:
            response = requests.post("http://localhost:5003/api/patient/ask",
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                answer = data.get('response', 'No response')
                print(f"Q: {question}")
                print(f"A: {answer}")
                print()
            else:
                print(f"Error asking: {question}")
                
    except Exception as e:
        print(f"Error testing PatientBot: {e}")
        print("Make sure PatientBot API is running: python API.py")

def main():
    """Main function for patient selection"""
    print("🏥 Patient Case Selection Tool")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show first 25 cases")
        print("2. Show first 50 cases") 
        print("3. Show specific case details")
        print("4. Test PatientBot with specific case")
        print("5. Run CHW workflow with specific case")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            df = show_csv_data(25)
        elif choice == "2":
            df = show_csv_data(50)
        elif choice == "3":
            df = pd.read_csv('data.csv')
            case_id = input(f"Enter case ID (0-{len(df)-1}): ").strip()
            if case_id.isdigit():
                show_detailed_case(df, int(case_id))
            else:
                print("Invalid case ID")
        elif choice == "4":
            df = pd.read_csv('data.csv')
            case_id = input(f"Enter case ID to test (0-{len(df)-1}): ").strip()
            if case_id.isdigit():
                test_patientbot_with_case(int(case_id))
            else:
                print("Invalid case ID")
        elif choice == "5":
            df = pd.read_csv('data.csv')
            case_id = input(f"Enter case ID for CHW workflow (0-{len(df)-1}): ").strip()
            if case_id.isdigit():
                print(f"\n🏥 Starting CHW workflow with Case {case_id}...")
                print("Run: python chw.py")
                print(f"Then select case ID: {case_id}")
                break
            else:
                print("Invalid case ID")
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 