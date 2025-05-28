#!/usr/bin/env python3
"""
Full ChatCHW Integration Test with PatientBot
Tests the complete workflow: PatientBot -> CHW Assessment -> MATRIX AI -> Diagnosis
"""

import requests
import json
import sys
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class FullCHWTester:
    def __init__(self):
        self.patient_bot_url = "http://localhost:5003"
        self.chw_api_url = "http://localhost:5001"  # Main CHW API
        self.workflow_api_url = "http://localhost:5002"  # Workflow API
        self.outputs_dir = "outputs"
        self.current_test_session = None
        
        # Load patient data for case selection
        self.df = pd.read_csv('data.csv')
        
    def display_available_cases(self, limit=20):
        """Display available patient cases for selection"""
        print("\n📋 AVAILABLE PATIENT CASES:")
        print("=" * 80)
        print(f"{'ID':<4} {'Age':<4} {'Sex':<8} {'Complaint':<15} {'Duration':<12} {'Preview'}")
        print("-" * 80)
        
        for i in range(min(limit, len(self.df))):
            row = self.df.iloc[i]
            preview = str(row['CHW Questions'])[:30] + "..." if len(str(row['CHW Questions'])) > 30 else str(row['CHW Questions'])
            print(f"{i:<4} {int(row['age (years)']):<4} {row['Sex']:<8} {row['Complaint']:<15} {row['Duration']:<12} {preview}")
        
        if len(self.df) > limit:
            print(f"\n... and {len(self.df) - limit} more cases available")
        print(f"\nTotal cases available: {len(self.df)}")
    
    def select_patient_case(self):
        """Allow user to select a patient case"""
        while True:
            try:
                self.display_available_cases()
                print(f"\n🎯 SELECT PATIENT CASE:")
                print(f"Enter case ID (0-{len(self.df)-1}) or 'random' for random case:")
                
                choice = input("Case ID: ").strip().lower()
                
                if choice == 'random':
                    import random
                    case_id = random.randint(0, len(self.df) - 1)
                    print(f"🎲 Selected random case: {case_id}")
                    return case_id
                elif choice.isdigit():
                    case_id = int(choice)
                    if 0 <= case_id < len(self.df):
                        return case_id
                    else:
                        print(f"❌ Invalid case ID. Must be between 0 and {len(self.df)-1}")
                else:
                    print("❌ Invalid input. Enter a number or 'random'")
            except KeyboardInterrupt:
                print("\n\n⏹️ Test cancelled by user")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def load_patient_case(self, case_id):
        """Load a patient case in PatientBot"""
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/load", 
                                   json={"case_id": case_id})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['patient']
                else:
                    print(f"❌ Error loading patient: {data.get('message')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to PatientBot API! Make sure it's running on port 5003")
        except Exception as e:
            print(f"❌ Error: {e}")
        return None
    
    def ask_patient(self, question):
        """Ask the patient a question via PatientBot"""
        try:
            response = requests.post(f"{self.patient_bot_url}/api/patient/ask",
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data['response']
                else:
                    print(f"❌ Error asking question: {data.get('message')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"❌ Error asking question: {e}")
        return "Error: Could not get response"
    
    def run_chw_assessment(self, patient_info, conversation_log):
        """Run CHW assessment using the main API"""
        try:
            # Prepare assessment data
            assessment_data = {
                "patient_info": patient_info,
                "conversation": conversation_log,
                "assessment_type": "full_workflow"
            }
            
            response = requests.post(f"{self.chw_api_url}/api/health/assessment",
                                   json=assessment_data)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ CHW Assessment Error: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to CHW API! Make sure it's running on port 5001")
        except Exception as e:
            print(f"❌ CHW Assessment Error: {e}")
        return None
    
    def run_workflow_assessment(self, patient_info):
        """Run workflow assessment using the workflow API"""
        try:
            # Start workflow session
            response = requests.post(f"{self.workflow_api_url}/api/workflow/start")
            if response.status_code != 200:
                print(f"❌ Workflow start error: {response.status_code}")
                return None
            
            session_data = response.json()
            session_id = session_data['session_id']
            
            # Simulate workflow responses
            workflow_log = []
            current_phase = "initial"
            
            while current_phase != "complete":
                # Get current session state
                response = requests.get(f"{self.workflow_api_url}/api/workflow/session/{session_id}")
                if response.status_code != 200:
                    break
                
                session_state = response.json()
                current_question = session_state.get('current_question')
                current_phase = session_state.get('phase', 'complete')
                
                if current_phase == "complete":
                    break
                
                if current_question:
                    # Ask PatientBot the question
                    patient_response = self.ask_patient(current_question)
                    
                    # Send response to workflow
                    response = requests.post(f"{self.workflow_api_url}/api/workflow/respond",
                                           json={
                                               "session_id": session_id,
                                               "response": patient_response
                                           })
                    
                    workflow_log.append({
                        "question": current_question,
                        "response": patient_response,
                        "phase": current_phase
                    })
                    
                    if response.status_code != 200:
                        break
            
            # Get final assessment
            response = requests.get(f"{self.workflow_api_url}/api/workflow/session/{session_id}")
            if response.status_code == 200:
                final_state = response.json()
                return {
                    "workflow_log": workflow_log,
                    "final_assessment": final_state
                }
            
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Workflow API! Make sure it's running on port 5002")
        except Exception as e:
            print(f"❌ Workflow Error: {e}")
        return None
    
    def simulate_chw_conversation(self, patient_info):
        """Simulate a realistic CHW conversation"""
        print("\n🩺 SIMULATING CHW CONVERSATION...")
        print("=" * 60)
        
        # Standard CHW questions
        chw_questions = [
            "Hello, what brings you here today?",
            "How long have you been feeling this way?",
            "Can you describe your symptoms in more detail?",
            "Have you had any fever?",
            "Are you eating and drinking normally?",
            "Have you tried any treatments?",
            "Do you have any other symptoms?",
            "On a scale of 1-10, how would you rate your discomfort?",
            "Is this getting better, worse, or staying the same?"
        ]
        
        conversation_log = []
        
        for i, question in enumerate(chw_questions, 1):
            print(f"\n[{i}] 🩺 CHW: {question}")
            
            # Get patient response
            response = self.ask_patient(question)
            print(f"    🤒 Patient: {response}")
            
            conversation_log.append({
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Add small delay for realism
            import time
            time.sleep(0.5)
        
        return conversation_log
    
    def save_test_results(self, case_id, patient_info, conversation_log, assessments):
        """Save test results to output file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chw_test_case_{case_id}_{timestamp}.json"
        filepath = os.path.join(self.outputs_dir, filename)
        
        test_results = {
            "test_info": {
                "case_id": case_id,
                "timestamp": timestamp,
                "patient_info": patient_info
            },
            "conversation_log": conversation_log,
            "assessments": assessments,
            "summary": {
                "total_questions": len(conversation_log),
                "patient_responses": len([q for q in conversation_log if q['response'] != "other: Information not known"]),
                "unknown_responses": len([q for q in conversation_log if q['response'] == "other: Information not known"])
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Test results saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def display_test_summary(self, patient_info, conversation_log, assessments):
        """Display a summary of the test results"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        print(f"\n👤 PATIENT INFO:")
        print(f"   Age: {patient_info['age_years']} years ({patient_info['age_category']})")
        print(f"   Sex: {patient_info['sex']}")
        print(f"   Complaint: {patient_info['primary_complaint']}")
        print(f"   Duration: {patient_info['duration']}")
        
        print(f"\n💬 CONVERSATION STATS:")
        total_questions = len(conversation_log)
        known_responses = len([q for q in conversation_log if q['response'] != "other: Information not known"])
        unknown_responses = total_questions - known_responses
        
        print(f"   Total Questions: {total_questions}")
        print(f"   Known Responses: {known_responses} ({known_responses/total_questions*100:.1f}%)")
        print(f"   Unknown Responses: {unknown_responses} ({unknown_responses/total_questions*100:.1f}%)")
        
        print(f"\n🔍 ASSESSMENTS:")
        for assessment_type, result in assessments.items():
            if result:
                print(f"   {assessment_type}: ✅ Completed")
            else:
                print(f"   {assessment_type}: ❌ Failed")
    
    def run_full_test(self):
        """Run the complete CHW test workflow"""
        print("🤖 FULL CHW INTEGRATION TEST WITH PATIENTBOT")
        print("=" * 80)
        print("This test will:")
        print("1. Let you select a patient case")
        print("2. Load the patient in PatientBot")
        print("3. Simulate a CHW conversation")
        print("4. Run CHW assessments with MATRIX AI")
        print("5. Save results to outputs folder")
        print("=" * 80)
        
        # Step 1: Select patient case
        case_id = self.select_patient_case()
        
        # Step 2: Load patient
        print(f"\n🔄 Loading patient case {case_id}...")
        patient_info = self.load_patient_case(case_id)
        if not patient_info:
            print("❌ Failed to load patient case")
            return False
        
        print(f"✅ Loaded: {patient_info['age_years']} {patient_info['sex']} with {patient_info['primary_complaint']}")
        
        # Step 3: Simulate CHW conversation
        conversation_log = self.simulate_chw_conversation(patient_info)
        
        # Step 4: Run assessments
        print(f"\n🔬 RUNNING ASSESSMENTS...")
        assessments = {}
        
        # Try CHW API assessment
        print("   Running CHW API assessment...")
        chw_result = self.run_chw_assessment(patient_info, conversation_log)
        assessments['chw_api'] = chw_result
        
        # Try Workflow API assessment
        print("   Running Workflow API assessment...")
        workflow_result = self.run_workflow_assessment(patient_info)
        assessments['workflow_api'] = workflow_result
        
        # Step 5: Save results
        print(f"\n💾 SAVING RESULTS...")
        output_file = self.save_test_results(case_id, patient_info, conversation_log, assessments)
        
        # Step 6: Display summary
        self.display_test_summary(patient_info, conversation_log, assessments)
        
        print(f"\n🎉 TEST COMPLETED!")
        if output_file:
            print(f"📁 Results saved to: {output_file}")
        
        return True

def main():
    """Main function"""
    print("Starting Full CHW Integration Test...")
    print("Make sure the following APIs are running:")
    print("  - PatientBot API (port 5003): python API.py")
    print("  - CHW API (port 5001): python API.py")
    print("  - Workflow API (port 5002): python workflow_API.py")
    print()
    
    tester = FullCHWTester()
    
    try:
        success = tester.run_full_test()
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed!")
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")

if __name__ == "__main__":
    main() 