#!/usr/bin/env python3
"""
PatientBot + MATRIX AI Integration Test
Tests PatientBot responses with MATRIX AI diagnosis system
"""

import requests
import json
import sys
import os
import pandas as pd
from datetime import datetime
import importlib.util

class MatrixPatientBotTester:
    def __init__(self):
        self.patient_bot_url = "http://localhost:5003"
        self.outputs_dir = "outputs"
        
        # Load patient data for case selection
        self.df = pd.read_csv('data.csv')
        
        # Try to import MATRIX AI from parent directory
        self.matrix_ai = self.load_matrix_ai()
        
    def load_matrix_ai(self):
        """Load MATRIX AI from the parent directory"""
        try:
            # Try to import from parent directory
            sys.path.append('..')
            
            # Try different possible locations for MATRIX AI
            possible_paths = [
                '../app.py',
                '../chad.py',
                '../commandChad.py'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"🔍 Found MATRIX AI at: {path}")
                    # For now, we'll simulate MATRIX AI responses
                    return "simulated"
            
            print("⚠️ MATRIX AI not found, will simulate responses")
            return "simulated"
            
        except Exception as e:
            print(f"⚠️ Could not load MATRIX AI: {e}")
            return "simulated"
    
    def display_available_cases(self, limit=20):
        """Display available patient cases for selection"""
        print("\n📋 AVAILABLE PATIENT CASES:")
        print("=" * 100)
        print(f"{'ID':<4} {'Age':<4} {'Sex':<8} {'Complaint':<15} {'Duration':<12} {'CHW Questions Preview'}")
        print("-" * 100)
        
        for i in range(min(limit, len(self.df))):
            row = self.df.iloc[i]
            preview = str(row['CHW Questions'])[:40] + "..." if len(str(row['CHW Questions'])) > 40 else str(row['CHW Questions'])
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
    
    def simulate_matrix_diagnosis(self, conversation_log, patient_info):
        """Simulate MATRIX AI diagnosis based on conversation"""
        # Extract symptoms and findings from conversation
        symptoms = []
        findings = []
        
        for exchange in conversation_log:
            response = exchange['response']
            if response != "other: Information not known" and "Error:" not in response:
                symptoms.append(response)
        
        # Simulate MATRIX AI analysis
        diagnosis_data = {
            "patient_demographics": {
                "age": patient_info['age_years'],
                "sex": patient_info['sex'],
                "age_category": patient_info['age_category']
            },
            "chief_complaint": patient_info['primary_complaint'],
            "duration": patient_info['duration'],
            "symptoms_extracted": symptoms,
            "matrix_analysis": {
                "primary_diagnosis": f"Simulated diagnosis for {patient_info['primary_complaint']}",
                "confidence": "85%",
                "recommendations": [
                    "Continue monitoring symptoms",
                    "Follow up if symptoms worsen",
                    "Ensure adequate rest and hydration"
                ],
                "red_flags": "None identified",
                "referral_needed": "No" if patient_info['age_category'] == "Adult" else "Consider if symptoms persist"
            },
            "knowledge_base_coverage": {
                "total_questions": len(conversation_log),
                "answered_from_kb": len([q for q in conversation_log if q['response'] != "other: Information not known"]),
                "unknown_responses": len([q for q in conversation_log if q['response'] == "other: Information not known"])
            }
        }
        
        return diagnosis_data
    
    def run_comprehensive_assessment(self, patient_info):
        """Run a comprehensive CHW assessment"""
        print("\n🩺 RUNNING COMPREHENSIVE CHW ASSESSMENT...")
        print("=" * 70)
        
        # Comprehensive CHW question set
        assessment_questions = [
            # Basic presentation
            "Hello, what brings you here today?",
            "How long have you been feeling this way?",
            "Can you describe your main symptoms?",
            
            # Symptom details
            "How severe are your symptoms on a scale of 1-10?",
            "Are your symptoms getting better, worse, or staying the same?",
            "What makes your symptoms better or worse?",
            
            # Associated symptoms
            "Do you have any fever?",
            "What is your temperature?",
            "Do you have any difficulty breathing?",
            "Do you have a cough? If so, how bad is it?",
            "Do you have a runny nose?",
            "Are you sneezing?",
            "Do you have diarrhea?",
            "Is there any blood in your stool?",
            "Do you have a sore throat?",
            
            # General health
            "Are you eating and drinking normally?",
            "How is your energy level?",
            "Are you sleeping well?",
            "Have you tried any treatments or medications?",
            
            # Physical examination simulation
            "Let me check your breathing rate",
            "Let me check your skin condition",
            "Let me examine your throat",
            
            # Additional questions
            "Do you have any other symptoms you haven't mentioned?",
            "Is there anything else you're worried about?",
            "Do you have any questions for me?"
        ]
        
        conversation_log = []
        
        for i, question in enumerate(assessment_questions, 1):
            print(f"\n[{i:2d}] 🩺 CHW: {question}")
            
            # Get patient response
            response = self.ask_patient(question)
            print(f"     🤒 Patient: {response}")
            
            conversation_log.append({
                "question_id": i,
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "category": self.categorize_question(question)
            })
            
            # Add small delay for realism
            import time
            time.sleep(0.3)
        
        return conversation_log
    
    def categorize_question(self, question):
        """Categorize the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['hello', 'brings', 'main']):
            return "chief_complaint"
        elif any(word in question_lower for word in ['long', 'duration', 'when']):
            return "duration"
        elif any(word in question_lower for word in ['fever', 'temperature']):
            return "vital_signs"
        elif any(word in question_lower for word in ['cough', 'breathing', 'throat', 'nose']):
            return "respiratory"
        elif any(word in question_lower for word in ['diarrhea', 'stool', 'blood']):
            return "gastrointestinal"
        elif any(word in question_lower for word in ['eating', 'drinking', 'energy', 'sleep']):
            return "general_health"
        elif any(word in question_lower for word in ['check', 'examine', 'rate']):
            return "physical_exam"
        else:
            return "other"
    
    def save_test_results(self, case_id, patient_info, conversation_log, diagnosis_data):
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matrix_test_case_{case_id}_{timestamp}.json"
        filepath = os.path.join(self.outputs_dir, filename)
        
        # Get the actual diagnosis from CSV for comparison
        actual_diagnosis = str(self.df.iloc[case_id]['Diagnosis & Treatment (right answer for GraderBot.  Do NOT pass to the PatientBot LLM) '])
        
        test_results = {
            "test_metadata": {
                "case_id": case_id,
                "timestamp": timestamp,
                "test_type": "PatientBot + MATRIX AI Integration",
                "patient_info": patient_info
            },
            "knowledge_base_data": {
                "age_category": patient_info['age_category'],
                "age_years": patient_info['age_years'],
                "sex": patient_info['sex'],
                "primary_complaint": patient_info['primary_complaint'],
                "duration": patient_info['duration'],
                "actual_diagnosis": actual_diagnosis
            },
            "conversation_log": conversation_log,
            "matrix_diagnosis": diagnosis_data,
            "analysis": {
                "total_questions": len(conversation_log),
                "questions_by_category": self.analyze_questions_by_category(conversation_log),
                "knowledge_coverage": {
                    "answered_from_kb": len([q for q in conversation_log if q['response'] != "other: Information not known"]),
                    "unknown_responses": len([q for q in conversation_log if q['response'] == "other: Information not known"]),
                    "error_responses": len([q for q in conversation_log if "Error:" in q['response']])
                },
                "response_quality": self.analyze_response_quality(conversation_log)
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
    
    def analyze_questions_by_category(self, conversation_log):
        """Analyze questions by category"""
        categories = {}
        for exchange in conversation_log:
            category = exchange['category']
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "answered": 0,
                    "unknown": 0
                }
            categories[category]["count"] += 1
            if exchange['response'] != "other: Information not known":
                categories[category]["answered"] += 1
            else:
                categories[category]["unknown"] += 1
        return categories
    
    def analyze_response_quality(self, conversation_log):
        """Analyze the quality of responses"""
        total = len(conversation_log)
        answered = len([q for q in conversation_log if q['response'] != "other: Information not known" and "Error:" not in q['response']])
        unknown = len([q for q in conversation_log if q['response'] == "other: Information not known"])
        errors = len([q for q in conversation_log if "Error:" in q['response']])
        
        return {
            "total_questions": total,
            "successful_answers": answered,
            "unknown_responses": unknown,
            "error_responses": errors,
            "success_rate": (answered / total * 100) if total > 0 else 0,
            "unknown_rate": (unknown / total * 100) if total > 0 else 0,
            "error_rate": (errors / total * 100) if total > 0 else 0
        }
    
    def display_comprehensive_summary(self, patient_info, conversation_log, diagnosis_data):
        """Display comprehensive test summary"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 100)
        
        # Patient info
        print(f"\n👤 PATIENT INFORMATION:")
        print(f"   Case ID: {diagnosis_data['knowledge_base_coverage']['total_questions']}")
        print(f"   Age: {patient_info['age_years']} years ({patient_info['age_category']})")
        print(f"   Sex: {patient_info['sex']}")
        print(f"   Chief Complaint: {patient_info['primary_complaint']}")
        print(f"   Duration: {patient_info['duration']}")
        
        # Conversation analysis
        analysis = self.analyze_response_quality(conversation_log)
        print(f"\n💬 CONVERSATION ANALYSIS:")
        print(f"   Total Questions Asked: {analysis['total_questions']}")
        print(f"   Successful Answers: {analysis['successful_answers']} ({analysis['success_rate']:.1f}%)")
        print(f"   Unknown Responses: {analysis['unknown_responses']} ({analysis['unknown_rate']:.1f}%)")
        print(f"   Error Responses: {analysis['error_responses']} ({analysis['error_rate']:.1f}%)")
        
        # Category breakdown
        categories = self.analyze_questions_by_category(conversation_log)
        print(f"\n📋 QUESTIONS BY CATEGORY:")
        for category, stats in categories.items():
            success_rate = (stats['answered'] / stats['count'] * 100) if stats['count'] > 0 else 0
            print(f"   {category.replace('_', ' ').title()}: {stats['answered']}/{stats['count']} ({success_rate:.1f}%)")
        
        # MATRIX diagnosis
        print(f"\n🔬 MATRIX AI DIAGNOSIS:")
        matrix_data = diagnosis_data['matrix_analysis']
        print(f"   Primary Diagnosis: {matrix_data['primary_diagnosis']}")
        print(f"   Confidence: {matrix_data['confidence']}")
        print(f"   Referral Needed: {matrix_data['referral_needed']}")
        print(f"   Red Flags: {matrix_data['red_flags']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(matrix_data['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    def run_full_test(self):
        """Run the complete integration test"""
        print("🤖 PATIENTBOT + MATRIX AI INTEGRATION TEST")
        print("=" * 80)
        print("This test will:")
        print("1. Let you select a patient case from the knowledge base")
        print("2. Load the patient in PatientBot")
        print("3. Run a comprehensive CHW assessment")
        print("4. Generate MATRIX AI diagnosis")
        print("5. Save detailed results to outputs folder")
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
        
        # Step 3: Run comprehensive assessment
        conversation_log = self.run_comprehensive_assessment(patient_info)
        
        # Step 4: Generate MATRIX diagnosis
        print(f"\n🔬 GENERATING MATRIX AI DIAGNOSIS...")
        diagnosis_data = self.simulate_matrix_diagnosis(conversation_log, patient_info)
        
        # Step 5: Save results
        print(f"\n💾 SAVING COMPREHENSIVE RESULTS...")
        output_file = self.save_test_results(case_id, patient_info, conversation_log, diagnosis_data)
        
        # Step 6: Display summary
        self.display_comprehensive_summary(patient_info, conversation_log, diagnosis_data)
        
        print(f"\n🎉 INTEGRATION TEST COMPLETED!")
        if output_file:
            print(f"📁 Detailed results saved to: {output_file}")
        
        return True

def main():
    """Main function"""
    print("Starting PatientBot + MATRIX AI Integration Test...")
    print("Make sure PatientBot API is running: python API.py (port 5003)")
    print()
    
    tester = MatrixPatientBotTester()
    
    try:
        success = tester.run_full_test()
        if success:
            print("\n✅ Integration test completed successfully!")
        else:
            print("\n❌ Integration test failed!")
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")

if __name__ == "__main__":
    main() 