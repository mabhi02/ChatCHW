from flask import Flask, request, jsonify
import pandas as pd
import random
import re
import os
from typing import Dict, Any, Optional

app = Flask(__name__)

class KnowledgeBasePatientBot:
    def __init__(self, csv_path):
        """Initialize the PatientBot with the knowledge base from CSV"""
        self.df = pd.read_csv(csv_path)
        self.current_patient = None
        self.current_patient_data = {}
        
    def load_patient_case(self, case_id: int = None) -> Dict[str, Any]:
        """Load a specific patient case or random case from the knowledge base"""
        if case_id is None:
            case_id = random.randint(0, len(self.df) - 1)
        
        if case_id >= len(self.df):
            case_id = len(self.df) - 1
            
        patient_row = self.df.iloc[case_id]
        
        # Extract knowledge from columns A-G only
        self.current_patient_data = {
            'case_id': case_id,
            'age_category': str(patient_row['Age']),  # Column A
            'age_years': int(patient_row['age (years)']),  # Column B - convert to int
            'sex': str(patient_row['Sex']),  # Column C
            'complaint': str(patient_row['Complaint']),  # Column D
            'duration': str(patient_row['Duration']),  # Column E
            'chw_questions': str(patient_row['CHW Questions']),  # Column F
            'exam_findings': str(patient_row['Exam Findings'])  # Column G
        }
        
        return {
            'patient_id': case_id,
            'age_category': self.current_patient_data['age_category'],
            'age_years': self.current_patient_data['age_years'],
            'sex': self.current_patient_data['sex'],
            'primary_complaint': self.current_patient_data['complaint'],
            'duration': self.current_patient_data['duration']
        }
    
    def extract_from_chw_questions(self, search_terms: list) -> Optional[str]:
        """Extract specific information from CHW Questions field"""
        if not self.current_patient_data or 'chw_questions' not in self.current_patient_data:
            return None
            
        chw_data = str(self.current_patient_data['chw_questions']).lower()
        
        for term in search_terms:
            # Look for patterns like "term: value" or "term value"
            patterns = [
                rf"{term.lower()}:\s*([^;,\n]+)",
                rf"{term.lower()}\s*([^;,\n]+)",
                rf"{term.lower()}[\s:]*([^;,\n]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, chw_data)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = re.sub(r'^[:\s]+', '', value)
                    value = re.sub(r'[;\s]+$', '', value)
                    return value.title()
        
        return None
    
    def extract_from_exam_findings(self, search_terms: list) -> Optional[str]:
        """Extract specific information from Exam Findings field"""
        if not self.current_patient_data or 'exam_findings' not in self.current_patient_data:
            return None
            
        exam_data = str(self.current_patient_data['exam_findings']).lower()
        
        for term in search_terms:
            patterns = [
                rf"{term.lower()}:\s*([^;,\n]+)",
                rf"{term.lower()}\s*([^;,\n]+)",
                rf"{term.lower()}[\s:]*([^;,\n]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, exam_data)
                if match:
                    value = match.group(1).strip()
                    value = re.sub(r'^[:\s]+', '', value)
                    value = re.sub(r'[;\s]+$', '', value)
                    return value.title()
        
        return None
    
    def respond_to_question(self, question: str) -> str:
        """Respond to questions using ONLY the knowledge base data"""
        if not self.current_patient_data:
            return "No patient case loaded. Please load a patient case first."
        
        question_lower = question.lower()
        
        # Basic demographic questions
        if any(word in question_lower for word in ['age', 'old', 'years']):
            if 'how old' in question_lower or 'age' in question_lower:
                return f"{self.current_patient_data['age_years']} years old"
        
        if any(word in question_lower for word in ['sex', 'gender', 'male', 'female']):
            return self.current_patient_data['sex']
        
        if any(word in question_lower for word in ['complaint', 'problem', 'wrong', 'matter']):
            return f"{self.current_patient_data['complaint']}"
        
        if any(word in question_lower for word in ['duration', 'long', 'when', 'started']):
            return self.current_patient_data['duration']
        
        # Symptom-specific questions
        if any(word in question_lower for word in ['cough', 'coughing']):
            # Check severity/type
            if any(word in question_lower for word in ['bad', 'severe', 'mild', 'type']):
                result = self.extract_from_chw_questions(['cough'])
                if result:
                    return result
            # General cough question
            result = self.extract_from_chw_questions(['cough'])
            if result:
                return f"Yes, {result.lower()}"
            elif 'cough' in self.current_patient_data['complaint'].lower():
                return "Yes"
        
        if any(word in question_lower for word in ['breathing', 'breath', 'difficulty breathing']):
            result = self.extract_from_chw_questions(['difficulty breathing', 'breathing'])
            if result:
                return result
            
        if any(word in question_lower for word in ['runny nose', 'nose']):
            result = self.extract_from_chw_questions(['runny nose'])
            if result:
                return result
                
        if any(word in question_lower for word in ['sneezing', 'sneeze']):
            result = self.extract_from_chw_questions(['sneezing'])
            if result:
                return result
        
        if any(word in question_lower for word in ['fever', 'temperature']):
            result = self.extract_from_exam_findings(['temperature'])
            if result:
                return f"Temperature is {result}"
                
        if any(word in question_lower for word in ['diarrhea', 'diarrhoea']):
            if 'diarrhea' in self.current_patient_data['complaint'].lower():
                return "Yes"
            
        if any(word in question_lower for word in ['blood', 'stool']):
            result = self.extract_from_chw_questions(['blood in stool'])
            if result:
                return result
                
        if any(word in question_lower for word in ['dehydrat', 'sunken eyes', 'eyes']):
            result = self.extract_from_chw_questions(['sunken eyes'])
            if result:
                return f"Sunken eyes: {result}"
                
        if any(word in question_lower for word in ['drinking', 'drink']):
            result = self.extract_from_chw_questions(['drinking normally', 'drinks eagerly', 'unable to drink'])
            if result:
                return result
                
        if any(word in question_lower for word in ['lethargic', 'alert', 'conscious']):
            result = self.extract_from_chw_questions(['general condition', 'lethargic', 'alert'])
            if result:
                return result
                
        if any(word in question_lower for word in ['respiratory rate', 'breathing rate']):
            result = self.extract_from_exam_findings(['respiratory rate'])
            if result:
                return f"Respiratory rate: {result}"
                
        if any(word in question_lower for word in ['skin pinch', 'skin']):
            result = self.extract_from_exam_findings(['skin pinch'])
            if result:
                return f"Skin pinch: {result}"
                
        if any(word in question_lower for word in ['throat', 'sore throat']):
            if 'sore throat' in self.current_patient_data['complaint'].lower():
                return "Yes, very sore throat"
            result = self.extract_from_chw_questions(['very sore throat'])
            if result:
                return result
                
        if any(word in question_lower for word in ['white spots', 'tonsils', 'spots']):
            result = self.extract_from_chw_questions(['white spots on tonsils'])
            if result:
                return result
        
        # Default response when information is not in knowledge base
        return "other: Information not known"

# Initialize the bot
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'data.csv')
patient_bot = KnowledgeBasePatientBot(csv_path)

@app.route('/api/patient/load', methods=['POST'])
def load_patient():
    """Load a patient case from the knowledge base"""
    try:
        data = request.get_json() or {}
        case_id = data.get('case_id')
        
        patient_info = patient_bot.load_patient_case(case_id)
        
        return jsonify({
            'status': 'success',
            'patient': patient_info,
            'message': f'Loaded patient case {patient_info["patient_id"]}'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error loading patient: {str(e)}'
        }), 500

@app.route('/api/patient/ask', methods=['POST'])
def ask_patient():
    """Ask the patient a question"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Question is required'
            }), 400
        
        question = data['question']
        response = patient_bot.respond_to_question(question)
        
        return jsonify({
            'status': 'success',
            'question': question,
            'response': response,
            'patient_id': patient_bot.current_patient_data.get('case_id') if patient_bot.current_patient_data else None
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.route('/api/patient/info', methods=['GET'])
def get_patient_info():
    """Get current patient information"""
    try:
        if not patient_bot.current_patient_data:
            return jsonify({
                'status': 'error',
                'message': 'No patient case loaded'
            }), 400
        
        # Return only basic info, not the full knowledge base
        return jsonify({
            'status': 'success',
            'patient': {
                'case_id': patient_bot.current_patient_data['case_id'],
                'age_category': patient_bot.current_patient_data['age_category'],
                'age_years': patient_bot.current_patient_data['age_years'],
                'sex': patient_bot.current_patient_data['sex'],
                'primary_complaint': patient_bot.current_patient_data['complaint'],
                'duration': patient_bot.current_patient_data['duration']
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting patient info: {str(e)}'
        }), 500

@app.route('/api/patient/cases', methods=['GET'])
def list_cases():
    """List available patient cases"""
    try:
        data = request.get_json() or {}
        limit = data.get('limit', 25)  # Default to 25 cases, allow override
        
        cases = []
        for i in range(min(limit, len(patient_bot.df))):
            row = patient_bot.df.iloc[i]
            cases.append({
                'case_id': i,
                'age_category': str(row['Age']),
                'age_years': int(row['age (years)']),
                'sex': str(row['Sex']),
                'complaint': str(row['Complaint']),
                'duration': str(row['Duration'])
            })
        
        return jsonify({
            'status': 'success',
            'total_cases': len(patient_bot.df),
            'showing': len(cases),
            'cases': cases
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error listing cases: {str(e)}'
        }), 500

@app.route('/api/patient/demo', methods=['POST'])
def demo_conversation():
    """Demonstrate a conversation with a patient"""
    try:
        data = request.get_json() or {}
        case_id = data.get('case_id')
        
        # Load patient case
        patient_info = patient_bot.load_patient_case(case_id)
        
        # Sample questions to demonstrate
        demo_questions = [
            "What is wrong with you?",
            "How old are you?",
            "Are you male or female?",
            "How long have you had this problem?",
            "How bad is your cough?",
            "Do you have difficulty breathing?",
            "Do you have a runny nose?",
            "Do you have fever?",
            "What is your temperature?"
        ]
        
        conversation = []
        for question in demo_questions:
            response = patient_bot.respond_to_question(question)
            conversation.append({
                'question': question,
                'response': response
            })
        
        return jsonify({
            'status': 'success',
            'patient': patient_info,
            'conversation': conversation
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error in demo: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'PatientBot Knowledge Base API',
        'cases_loaded': len(patient_bot.df) if hasattr(patient_bot, 'df') else 0
    })

if __name__ == '__main__':
    print("🤖 PatientBot Knowledge Base API Starting...")
    print(f"📊 Loaded {len(patient_bot.df)} patient cases from knowledge base")
    print("🔍 API will only use information from CSV columns A-G")
    print("❓ Unknown information will return: 'other: Information not known'")
    print("🌐 Available endpoints:")
    print("   POST /api/patient/load - Load a patient case")
    print("   POST /api/patient/ask - Ask patient a question")
    print("   GET  /api/patient/info - Get current patient info")
    print("   GET  /api/patient/cases - List available cases")
    print("   POST /api/patient/demo - Demo conversation")
    print("   GET  /health - Health check")
    
    app.run(debug=True, port=5003, host='0.0.0.0') 