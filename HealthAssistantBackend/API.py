from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
import sys
import os
import openai
import datetime
import uuid
from chad import (
    get_embedding_batch,
    vectorQuotesWithSource,
    process_with_matrix,
    get_diagnosis_and_treatment,
    parse_question_data,
    get_openai_completion
)

app = Flask(__name__)
frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
CORS(app, origins=["http://localhost:3000", frontend_url], supports_credentials=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Global storage for sessions
assessment_sessions = {}

# Survey questions that can be optionally included
SURVEY_QUESTIONS = [
    {
        "id": "age",
        "question": "What is your age?",
        "type": "number",
        "required": False
    },
    {
        "id": "gender", 
        "question": "What is your gender?",
        "type": "multiple_choice",
        "options": ["Male", "Female", "Non-binary", "Other", "Prefer not to say"],
        "required": False
    },
    {
        "id": "duration",
        "question": "How long have you been experiencing these symptoms?",
        "type": "multiple_choice", 
        "options": ["Less than 24 hours", "1-7 days", "1-4 weeks", "More than a month", "Not applicable"],
        "required": False
    },
    {
        "id": "severity",
        "question": "How would you rate the severity of your symptoms?",
        "type": "multiple_choice",
        "options": ["Mild", "Moderate", "Severe", "Very severe", "Not applicable"],
        "required": False
    },
    {
        "id": "previous_episodes",
        "question": "Have you experienced similar symptoms before?",
        "type": "multiple_choice",
        "options": ["Yes, frequently", "Yes, occasionally", "Yes, once before", "No, this is the first time"],
        "required": False
    },
    {
        "id": "medications",
        "question": "Are you currently taking any medications?",
        "type": "text",
        "required": False
    },
    {
        "id": "medical_history",
        "question": "Do you have any known medical conditions or allergies?",
        "type": "text", 
        "required": False
    }
]

class HealthAssessmentAPI:
    """Main class for handling health assessments with optional survey questions"""
    
    def __init__(self):
        self.pinecone_index = None
        self.pinecone_available = False
        
        # Try to initialize Pinecone index with fallback options
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone index with error handling"""
        # List of possible index names to try
        possible_indexes = ["final-asha", "who-guide-old", "healthcareassistantchw"]
        
        for index_name in possible_indexes:
            try:
                self.pinecone_index = pc.Index(index_name)
                self.pinecone_available = True
                print(f"✅ Connected to Pinecone index: {index_name}")
                return
            except Exception as e:
                print(f"⚠️ Failed to connect to index '{index_name}': {e}")
                continue
        
        print("❌ No Pinecone indexes available. API will work with reduced functionality.")
        self.pinecone_available = False
    
    def create_session(self, session_id: str = None) -> str:
        """Create a new assessment session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        assessment_sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.datetime.now().isoformat(),
            "primary_complaint": None,
            "survey_responses": {},
            "assessment_results": None,
            "status": "created"
        }
        
        return session_id
    
    def process_primary_complaint(self, session_id: str, complaint: str) -> Dict[str, Any]:
        """Process the main 'what is wrong with you' question"""
        if session_id not in assessment_sessions:
            raise ValueError("Session not found")
        
        session = assessment_sessions[session_id]
        session["primary_complaint"] = complaint
        session["status"] = "complaint_received"
        
        # Get embeddings for the complaint (only if Pinecone is available)
        if self.pinecone_available and self.pinecone_index:
            try:
                embeddings = get_embedding_batch([complaint])
                if embeddings and embeddings[0]:
                    # Search for relevant medical information
                    relevant_chunks = vectorQuotesWithSource(
                        embeddings[0], 
                        self.pinecone_index, 
                        top_k=5
                    )
                    session["relevant_chunks"] = relevant_chunks
                else:
                    session["relevant_chunks"] = []
            except Exception as e:
                print(f"Error getting embeddings or searching vector DB: {e}")
                session["relevant_chunks"] = []
        else:
            print("⚠️ Pinecone not available, skipping vector search")
            session["relevant_chunks"] = []
        
        return {
            "session_id": session_id,
            "status": "success",
            "message": "Primary complaint received and processed",
            "relevant_information_found": len(session.get("relevant_chunks", [])) > 0,
            "pinecone_available": self.pinecone_available
        }
    
    def add_survey_responses(self, session_id: str, survey_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Add survey question responses to the session"""
        if session_id not in assessment_sessions:
            raise ValueError("Session not found")
        
        session = assessment_sessions[session_id]
        
        # Validate survey responses against known questions
        validated_responses = {}
        for question_id, response in survey_responses.items():
            # Find the question definition
            question_def = next((q for q in SURVEY_QUESTIONS if q["id"] == question_id), None)
            if question_def:
                validated_responses[question_id] = {
                    "question": question_def["question"],
                    "answer": response,
                    "type": question_def["type"]
                }
        
        session["survey_responses"] = validated_responses
        session["status"] = "survey_completed"
        
        return {
            "session_id": session_id,
            "status": "success", 
            "message": f"Survey responses added for {len(validated_responses)} questions",
            "responses_count": len(validated_responses)
        }
    
    def generate_assessment(self, session_id: str) -> Dict[str, Any]:
        """Generate health assessment based on complaint and optional survey data"""
        if session_id not in assessment_sessions:
            raise ValueError("Session not found")
        
        session = assessment_sessions[session_id]
        
        if not session.get("primary_complaint"):
            raise ValueError("Primary complaint is required")
        
        # Prepare data for assessment
        complaint = session["primary_complaint"]
        survey_responses = session.get("survey_responses", {})
        relevant_chunks = session.get("relevant_chunks", [])
        
        # Create a structured input for the MATRIX system
        structured_input = [
            {
                "question": "What is wrong with you today?",
                "answer": complaint,
                "type": "FREE"
            }
        ]
        
        # Add survey responses as additional context
        for response_data in survey_responses.values():
            structured_input.append(response_data)
        
        try:
            # Use MATRIX processing for enhanced analysis (only if system is available)
            matrix_output = process_with_matrix(
                current_text=complaint,
                previous_responses=structured_input,
                context_text="\n".join([chunk["text"] for chunk in relevant_chunks[:3]])
            )
            
            # Generate comprehensive assessment
            assessment_result = self._generate_comprehensive_assessment(
                complaint, survey_responses, relevant_chunks, matrix_output
            )
            
            session["assessment_results"] = assessment_result
            session["matrix_analysis"] = matrix_output
            session["status"] = "completed"
            
            return {
                "session_id": session_id,
                "status": "success",
                "assessment": assessment_result,
                "confidence_score": matrix_output.get("confidence", 0.5),
                "recommendations": assessment_result.get("recommendations", []),
                "method": "matrix_enhanced",
                "pinecone_available": self.pinecone_available
            }
            
        except Exception as e:
            print(f"Error generating assessment with MATRIX: {e}")
            # Fallback to basic assessment
            basic_assessment = self._generate_basic_assessment(complaint, survey_responses, relevant_chunks)
            session["assessment_results"] = basic_assessment
            session["status"] = "completed_basic"
            
            return {
                "session_id": session_id,
                "status": "success",
                "assessment": basic_assessment,
                "confidence_score": 0.3,
                "note": "Used fallback assessment method due to system unavailability",
                "method": "basic_fallback",
                "pinecone_available": self.pinecone_available
            }
    
    def _generate_comprehensive_assessment(self, complaint: str, survey_responses: Dict, 
                                         relevant_chunks: List, matrix_output: Dict) -> Dict[str, Any]:
        """Generate comprehensive assessment using MATRIX output"""
        
        # Build context from survey responses
        survey_context = ""
        if survey_responses:
            survey_context = "\nAdditional patient information:\n"
            for resp_data in survey_responses.values():
                survey_context += f"- {resp_data['question']}: {resp_data['answer']}\n"
        
        # Build context from relevant medical information
        medical_context = ""
        if relevant_chunks:
            medical_context = "\nRelevant medical information:\n"
            for chunk in relevant_chunks[:3]:
                medical_context += f"- {chunk['text'][:200]}...\n"
        
        # Generate assessment prompt
        assessment_prompt = f"""
        As a healthcare assistant, provide a comprehensive assessment based on:
        
        Primary complaint: {complaint}
        {survey_context}
        {medical_context}
        
        MATRIX Analysis Confidence: {matrix_output.get('confidence', 'Not available')}
        
        Please provide:
        1. Possible conditions or diagnoses to consider
        2. Recommended immediate actions
        3. When to seek professional medical care
        4. General health advice
        5. Red flags to watch for
        
        Format as JSON with keys: possible_conditions, immediate_actions, seek_care, general_advice, red_flags
        """
        
        try:
            assessment_text = get_openai_completion(assessment_prompt, max_tokens=500, temperature=0.3)
            
            # Try to parse as JSON, fallback to structured text
            try:
                assessment_json = json.loads(assessment_text)
                return assessment_json
            except json.JSONDecodeError:
                return {
                    "assessment_text": assessment_text,
                    "recommendations": ["Consult with a healthcare professional for proper diagnosis"],
                    "confidence_level": "moderate"
                }
        except Exception as e:
            print(f"Error in comprehensive assessment: {e}")
            return self._generate_basic_assessment(complaint, survey_responses, relevant_chunks)
    
    def _generate_basic_assessment(self, complaint: str, survey_responses: Dict, 
                                 relevant_chunks: List) -> Dict[str, Any]:
        """Generate basic assessment without MATRIX processing"""
        
        recommendations = [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Consult with a healthcare professional if symptoms persist or worsen"
        ]
        
        # Add specific recommendations based on survey responses
        if survey_responses:
            if any("severe" in str(resp.get("answer", "")).lower() for resp in survey_responses.values()):
                recommendations.insert(0, "Seek immediate medical attention due to severe symptoms")
            
            age_resp = survey_responses.get("age", {})
            if age_resp and isinstance(age_resp.get("answer"), (int, str)):
                try:
                    age = int(age_resp["answer"])
                    if age > 65:
                        recommendations.append("Consider additional monitoring due to age-related factors")
                except (ValueError, TypeError):
                    pass
        
        return {
            "summary": f"Assessment for: {complaint[:100]}...",
            "recommendations": recommendations,
            "advice": "This is a preliminary assessment. Professional medical consultation is recommended.",
            "relevant_information": [chunk["text"][:100] + "..." for chunk in relevant_chunks[:2]]
        }

# Initialize the API handler
health_api = HealthAssessmentAPI()

@app.route('/api/health/session', methods=['POST'])
def create_assessment_session():
    """Create a new health assessment session"""
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        
        new_session_id = health_api.create_session(session_id)
        
        return jsonify({
            "status": "success",
            "session_id": new_session_id,
            "message": "Assessment session created"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/health/complaint', methods=['POST'])
def submit_complaint():
    """Submit the primary health complaint"""
    try:
        data = request.json
        session_id = data.get('session_id')
        complaint = data.get('complaint')
        
        if not session_id or not complaint:
            return jsonify({
                "status": "error",
                "message": "session_id and complaint are required"
            }), 400
        
        result = health_api.process_primary_complaint(session_id, complaint)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/survey', methods=['POST'])
def submit_survey():
    """Submit survey question responses"""
    try:
        data = request.json
        session_id = data.get('session_id')
        survey_responses = data.get('survey_responses', {})
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required"
            }), 400
        
        result = health_api.add_survey_responses(session_id, survey_responses)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/assessment', methods=['POST'])
def generate_health_assessment():
    """Generate health assessment based on complaint and survey data"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required"
            }), 400
        
        result = health_api.generate_assessment(session_id)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/complete-assessment', methods=['POST'])
def complete_assessment():
    """Complete assessment with both complaint and optional survey in one call"""
    try:
        data = request.json
        session_id = data.get('session_id')
        complaint = data.get('complaint')
        survey_responses = data.get('survey_responses', {})
        
        if not complaint:
            return jsonify({
                "status": "error", 
                "message": "complaint is required"
            }), 400
        
        # Create session if not provided
        if not session_id:
            session_id = health_api.create_session()
        
        # Process complaint
        health_api.process_primary_complaint(session_id, complaint)
        
        # Add survey responses if provided
        if survey_responses:
            health_api.add_survey_responses(session_id, survey_responses)
        
        # Generate assessment
        result = health_api.generate_assessment(session_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/session/<session_id>', methods=['GET'])
def get_session_info():
    """Get session information and status"""
    try:
        if session_id not in assessment_sessions:
            return jsonify({
                "status": "error",
                "message": "Session not found"
            }), 404
        
        session = assessment_sessions[session_id]
        return jsonify({
            "status": "success",
            "session": {
                "session_id": session_id,
                "created_at": session["created_at"],
                "status": session["status"],
                "has_complaint": session.get("primary_complaint") is not None,
                "survey_responses_count": len(session.get("survey_responses", {})),
                "has_assessment": session.get("assessment_results") is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/survey-questions', methods=['GET'])
def get_survey_questions():
    """Get available survey questions"""
    return jsonify({
        "status": "success",
        "survey_questions": SURVEY_QUESTIONS
    })

@app.route('/api/health/demo', methods=['POST'])
def demo_endpoint():
    """Demo endpoint showing how to use the API"""
    try:
        demo_complaint = "I have been having headaches for the past few days"
        demo_survey = {
            "age": "30",
            "gender": "Female", 
            "duration": "1-7 days",
            "severity": "Moderate"
        }
        
        # Create session and run complete assessment
        session_id = health_api.create_session()
        health_api.process_primary_complaint(session_id, demo_complaint)
        health_api.add_survey_responses(session_id, demo_survey)
        result = health_api.generate_assessment(session_id)
        
        return jsonify({
            "demo": True,
            "example_complaint": demo_complaint,
            "example_survey": demo_survey,
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health/status', methods=['GET'])
def get_api_status():
    """Get API health status and system availability"""
    try:
        # Check OpenAI API key
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        
        # Check Pinecone status
        pinecone_available = health_api.pinecone_available
        pinecone_index_name = None
        if pinecone_available and health_api.pinecone_index:
            try:
                # Try to get index stats to verify it's working
                stats = health_api.pinecone_index.describe_index_stats()
                pinecone_index_name = getattr(health_api.pinecone_index, '_index_name', 'Unknown')
            except:
                pinecone_available = False
        
        return jsonify({
            "status": "success",
            "api_version": "1.0.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "systems": {
                "openai": {
                    "available": openai_available,
                    "status": "✅ Available" if openai_available else "❌ API key not found"
                },
                "pinecone": {
                    "available": pinecone_available,
                    "index_name": pinecone_index_name,
                    "status": f"✅ Connected to {pinecone_index_name}" if pinecone_available else "❌ Not available"
                },
                "matrix": {
                    "available": True,  # Assume available since imported
                    "status": "✅ Available"
                }
            },
            "capabilities": {
                "basic_assessment": True,
                "enhanced_assessment": openai_available,
                "vector_search": pinecone_available,
                "matrix_processing": True
            },
            "endpoints": [
                "/api/health/session",
                "/api/health/complaint", 
                "/api/health/survey",
                "/api/health/assessment",
                "/api/health/complete-assessment",
                "/api/health/session/<id>",
                "/api/health/survey-questions",
                "/api/health/demo",
                "/api/health/status"
            ]
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 