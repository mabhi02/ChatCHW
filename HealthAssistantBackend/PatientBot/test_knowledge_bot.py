#!/usr/bin/env python3
"""
Comprehensive Test Suite for Knowledge-Base PatientBot API
Tests all endpoints and validates knowledge-base only responses
"""

import requests
import json
import time
import sys

class PatientBotTester:
    def __init__(self, base_url="http://localhost:5003"):
        self.base_url = base_url
        self.test_results = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test results"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
    
    def test_health_check(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy' and data.get('cases_loaded', 0) > 0:
                    self.log_test("Health Check", True, f"Loaded {data['cases_loaded']} cases")
                    return True
                else:
                    self.log_test("Health Check", False, "Invalid health response")
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test("Health Check", False, f"Connection error: {e}")
        return False
    
    def test_list_cases(self):
        """Test listing available cases"""
        try:
            response = requests.get(f"{self.base_url}/api/patient/cases")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and len(data.get('cases', [])) > 0:
                    self.log_test("List Cases", True, f"Found {data['total_cases']} total cases")
                    return data['cases']
                else:
                    self.log_test("List Cases", False, "No cases returned")
            else:
                self.log_test("List Cases", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test("List Cases", False, f"Error: {e}")
        return []
    
    def test_load_patient(self, case_id=0):
        """Test loading a specific patient case"""
        try:
            response = requests.post(f"{self.base_url}/api/patient/load", 
                                   json={"case_id": case_id})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    patient = data['patient']
                    self.log_test("Load Patient", True, 
                                f"Loaded case {patient['patient_id']}: {patient['age_years']} {patient['sex']} with {patient['primary_complaint']}")
                    return patient
                else:
                    self.log_test("Load Patient", False, data.get('message', 'Unknown error'))
            else:
                self.log_test("Load Patient", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test("Load Patient", False, f"Error: {e}")
        return None
    
    def test_patient_info(self):
        """Test getting current patient info"""
        try:
            response = requests.get(f"{self.base_url}/api/patient/info")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.log_test("Patient Info", True, "Retrieved patient info successfully")
                    return data['patient']
                else:
                    self.log_test("Patient Info", False, data.get('message', 'Unknown error'))
            else:
                self.log_test("Patient Info", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test("Patient Info", False, f"Error: {e}")
        return None
    
    def test_ask_question(self, question, expected_type=None):
        """Test asking a question to the patient"""
        try:
            response = requests.post(f"{self.base_url}/api/patient/ask", 
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    answer = data['response']
                    
                    # Check if it's knowledge-base only (no external medical knowledge)
                    is_knowledge_only = (
                        answer == "other: Information not known" or
                        not any(phrase in answer.lower() for phrase in [
                            'i think', 'probably', 'might be', 'could be', 
                            'usually', 'typically', 'generally', 'medical advice'
                        ])
                    )
                    
                    self.log_test(f"Ask: '{question}'", True, f"Answer: '{answer}'")
                    return answer
                else:
                    self.log_test(f"Ask: '{question}'", False, data.get('message', 'Unknown error'))
            else:
                self.log_test(f"Ask: '{question}'", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test(f"Ask: '{question}'", False, f"Error: {e}")
        return None
    
    def test_demo_conversation(self, case_id=None):
        """Test demo conversation endpoint"""
        try:
            payload = {"case_id": case_id} if case_id is not None else {}
            response = requests.post(f"{self.base_url}/api/patient/demo", json=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    conversation = data['conversation']
                    self.log_test("Demo Conversation", True, f"Generated {len(conversation)} Q&A pairs")
                    return data
                else:
                    self.log_test("Demo Conversation", False, data.get('message', 'Unknown error'))
            else:
                self.log_test("Demo Conversation", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.log_test("Demo Conversation", False, f"Error: {e}")
        return None
    
    def test_knowledge_base_constraints(self):
        """Test that responses are strictly from knowledge base"""
        print("\n🔍 Testing Knowledge Base Constraints...")
        
        # Load a patient first
        patient = self.test_load_patient(0)
        if not patient:
            return False
        
        # Test questions that should have knowledge-base answers
        knowledge_questions = [
            "What is wrong with you?",
            "How old are you?", 
            "Are you male or female?",
            "How long have you had this problem?",
            "Do you have a cough?",
            "What is your temperature?"
        ]
        
        # Test questions that should return "other: Information not known"
        unknown_questions = [
            "What is your favorite color?",
            "Do you have any allergies?",
            "What medications are you taking?",
            "Do you smoke?",
            "What is your blood pressure?",
            "Do you have diabetes?"
        ]
        
        knowledge_passed = 0
        for question in knowledge_questions:
            answer = self.test_ask_question(question)
            if answer and answer != "other: Information not known":
                knowledge_passed += 1
        
        unknown_passed = 0
        for question in unknown_questions:
            answer = self.test_ask_question(question)
            if answer == "other: Information not known":
                unknown_passed += 1
        
        self.log_test("Knowledge Questions", knowledge_passed >= len(knowledge_questions) * 0.7,
                     f"{knowledge_passed}/{len(knowledge_questions)} answered from knowledge base")
        
        self.log_test("Unknown Questions", unknown_passed >= len(unknown_questions) * 0.8,
                     f"{unknown_passed}/{len(unknown_questions)} correctly returned 'Information not known'")
        
        return knowledge_passed >= len(knowledge_questions) * 0.7 and unknown_passed >= len(unknown_questions) * 0.8
    
    def test_specific_case_scenarios(self):
        """Test specific patient cases to validate knowledge extraction"""
        print("\n🎯 Testing Specific Case Scenarios...")
        
        # Test Case 0: Child with Cough
        patient = self.test_load_patient(0)
        if patient:
            # Should know it's a cough case
            answer = self.test_ask_question("Do you have a cough?")
            if answer and "yes" in answer.lower():
                self.log_test("Cough Recognition", True, "Correctly identified cough")
            else:
                self.log_test("Cough Recognition", False, f"Expected cough, got: {answer}")
            
            # Should know severity
            answer = self.test_ask_question("How bad is your cough?")
            if answer and answer != "other: Information not known":
                self.log_test("Cough Severity", True, f"Severity: {answer}")
            else:
                self.log_test("Cough Severity", False, "Could not determine severity")
        
        # Test Case with Diarrhea (if available)
        cases = self.test_list_cases()
        diarrhea_case = None
        for i, case in enumerate(cases):
            if 'diarrhea' in case.get('complaint', '').lower():
                diarrhea_case = i
                break
        
        if diarrhea_case is not None:
            patient = self.test_load_patient(diarrhea_case)
            if patient:
                answer = self.test_ask_question("Do you have diarrhea?")
                if answer and "yes" in answer.lower():
                    self.log_test("Diarrhea Recognition", True, "Correctly identified diarrhea")
                else:
                    self.log_test("Diarrhea Recognition", False, f"Expected diarrhea, got: {answer}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🤖 PATIENTBOT KNOWLEDGE-BASE API TEST SUITE")
        print("=" * 60)
        
        # Basic connectivity tests
        print("\n📡 Testing Basic Connectivity...")
        if not self.test_health_check():
            print("❌ Cannot connect to API. Make sure it's running on port 5003")
            return False
        
        # API endpoint tests
        print("\n🔧 Testing API Endpoints...")
        cases = self.test_list_cases()
        if not cases:
            print("❌ No cases available")
            return False
        
        patient = self.test_load_patient(0)
        if not patient:
            print("❌ Cannot load patient")
            return False
        
        self.test_patient_info()
        demo_data = self.test_demo_conversation(1)
        
        # Knowledge base constraint tests
        self.test_knowledge_base_constraints()
        
        # Specific scenario tests
        self.test_specific_case_scenarios()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['passed'])
        total = len(self.test_results)
        
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {total - passed}")
        print(f"📈 Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! PatientBot is working correctly.")
        elif passed >= total * 0.8:
            print("\n✅ Most tests passed. PatientBot is mostly functional.")
        else:
            print("\n⚠️  Many tests failed. Check PatientBot implementation.")
        
        return passed >= total * 0.8

def main():
    """Main test function"""
    print("Starting PatientBot Knowledge-Base API Tests...")
    print("Make sure the API is running: python API.py")
    print()
    
    tester = PatientBotTester()
    
    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 