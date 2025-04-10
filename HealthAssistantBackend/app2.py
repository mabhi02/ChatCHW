from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import traceback
from typing import Dict, List, Any, Optional

# Import all needed functions from chad2
from chad2 import (
    setup_database,
    create_or_get_session,
    update_session_responses,
    update_session_phase,
    update_session_field,
    get_session_responses,
    get_session_field,
    get_structured_questions,
    get_examination_history,
    generate_question_with_options,
    judge,
    judge_exam,
    generate_examination,
    get_diagnosis_and_treatment,
    handle_initial_question,
    handle_followup_response,
    handle_examination_response,
    questions_init,
    init
)

app = Flask(__name__)
# Set frontend URL with fallback to your deployed frontend
frontend_url = os.environ.get('FRONTEND_URL', 'https://chatchw.onrender.com')
# Configure CORS to allow requests from both localhost and your deployed frontend
CORS(app, origins=["http://localhost:3000", frontend_url], supports_credentials=True)

# Initialize the application
init()

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new session and return first question"""
    try:
        # Get session ID from request
        data = request.json
        session_id = data.get('session_id', 'default')
        
        # Initialize or get existing session
        session = create_or_get_session(session_id)
        
        # Get the current question index
        current_question_index = session.get('current_question_index', 0)
        
        # Format the current question for the UI
        if current_question_index < len(questions_init):
            current_question = questions_init[current_question_index]
            
            # Format the output based on the question type
            if current_question['type'] in ['MC', 'MCM', 'YN']:
                options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in current_question['options']])
                output = f"{current_question['question']}\n\n{options_text}"
            else:
                output = current_question['question']
            
            return jsonify({
                "status": "success",
                "output": output,
                "metadata": {
                    "phase": session.get('phase', 'initial'),
                    "question_type": current_question['type'],
                    "options": current_question.get('options', []),
                    "question_index": current_question_index
                }
            })
        else:
            # If all initial questions are answered, move to followup phase
            update_session_phase(session_id, "followup")
            
            # Generate and return the first followup question
            return generate_followup_question(session_id)
            
    except Exception as e:
        print(f"Error starting assessment: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/input', methods=['POST'])
def process_input():
    """Process user input and return next question/response"""
    try:
        # Get data from the request
        data = request.json
        session_id = data.get('session_id', 'default')
        user_input = data.get('input', '')
        
        # Get the current session phase
        phase = get_session_field(session_id, "phase")
        
        print(f"Processing input for session {session_id} in phase {phase}")
        
        if phase == "initial":
            # Get the current question index
            current_question_index = get_session_field(session_id, "current_question_index")
            
            # Process the initial question response
            result = handle_initial_question(session_id, current_question_index, user_input)
            
            if result['status'] == "error":
                return jsonify(result)
            
            # Check if we've completed all initial questions
            if result['next_question_index'] >= len(questions_init):
                # Move to followup phase
                update_session_phase(session_id, "followup")
                return generate_followup_question(session_id)
            else:
                # Return the next initial question
                next_question = questions_init[result['next_question_index']]
                
                # Format the output based on the question type
                if next_question['type'] in ['MC', 'MCM', 'YN']:
                    options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                    output = f"{next_question['question']}\n\n{options_text}"
                else:
                    output = next_question['question']
                
                return jsonify({
                    "status": "success",
                    "output": output,
                    "metadata": {
                        "phase": "initial",
                        "question_type": next_question['type'],
                        "options": next_question.get('options', []),
                        "question_index": result['next_question_index']
                    }
                })
        
        elif phase == "followup":
            # Get the current followup question from the request
            current_question = data.get('current_question', {})
            
            if not current_question:
                return jsonify({
                    "status": "error",
                    "message": "Missing current question data"
                })
            
            # Process the followup response
            result = handle_followup_response(session_id, user_input, current_question)
            
            if result['status'] == "error":
                return jsonify(result)
            
            # Check if we should move to the examination phase
            if result['next_phase'] == "examination":
                return generate_examination_question(session_id)
            else:
                # Generate and return the next followup question
                return generate_followup_question(session_id)
        
        elif phase == "examination":
            # Get the current examination from the request
            current_examination = data.get('current_examination', {})
            
            if not current_examination:
                return jsonify({
                    "status": "error",
                    "message": "Missing current examination data"
                })
            
            # Process the examination response
            result = handle_examination_response(session_id, user_input, current_examination)
            
            if result['status'] == "error":
                return jsonify(result)
            
            # Check if we should move to the diagnosis phase
            if result['next_phase'] == "diagnosis":
                return generate_diagnosis(session_id)
            else:
                # Generate and return the next examination
                return generate_examination_question(session_id)
        
        elif phase == "diagnosis" or phase == "complete":
            # Return the diagnosis and treatment again if requested
            return generate_diagnosis(session_id)
        
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown phase: {phase}"
            })
    
    except Exception as e:
        print(f"Error processing input: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_followup_question(session_id: str):
    """Generate a follow-up question and return it."""
    try:
        # Generate the next question
        question_data = generate_question_with_options(session_id)
        
        # Format the output for the UI
        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in question_data['options']])
        output = f"{question_data['question']}\n\n{options_text}"
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "followup",
                "question_type": question_data['type'],
                "options": question_data['options'],
                "current_question": question_data
            }
        })
    
    except Exception as e:
        print(f"Error generating followup question: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_examination_question(session_id: str):
    """Generate an examination question and return it."""
    try:
        # Generate the examination
        examination_data = generate_examination(session_id)
        
        # Format the output for the UI
        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in examination_data['options']])
        output = f"Recommended Examination:\n{examination_data['examination']}\n\nFindings:\n{options_text}"
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "examination",
                "question_type": examination_data['type'],
                "options": examination_data['options'],
                "current_examination": examination_data
            }
        })
    
    except Exception as e:
        print(f"Error generating examination: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_diagnosis(session_id: str):
    """Generate and return diagnosis and treatment information."""
    try:
        # Check if the diagnosis has already been generated
        diagnosis = get_session_field(session_id, "diagnosis")
        treatment = get_session_field(session_id, "treatment")
        
        if not diagnosis or not treatment:
            # Generate new diagnosis and treatment
            results = get_diagnosis_and_treatment(session_id)
            diagnosis = results['diagnosis']
            treatment = results['treatment']
            citations = results['citations']
        else:
            # Query existing references from the database
            # This is simplified - in a real implementation you'd get these from the database
            citations = []
        
        # Format the output for the UI
        output = f"""Assessment Complete

Diagnosis:
{diagnosis}

Treatment Plan:
{treatment}

Key References:
{chr(10).join([f"- {cite['source']} (relevance: {cite['score']:.2f})" for cite in citations]) if citations else "No references available."}"""
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "complete",
                "diagnosis": diagnosis,
                "treatment": treatment,
                "references": citations
            }
        })
    
    except Exception as e:
        print(f"Error generating diagnosis: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get the current state of a session."""
    try:
        # Retrieve the session from the database
        session = create_or_get_session(session_id)
        
        if not session:
            return jsonify({
                "status": "error",
                "message": f"Session {session_id} not found"
            })
        
        # Get structured questions and examination history
        questions = get_structured_questions(session_id)
        examinations = get_examination_history(session_id)
        
        # Return the session data
        return jsonify({
            "status": "success",
            "session": session,
            "questions": questions,
            "examinations": examinations
        })
    
    except Exception as e:
        print(f"Error retrieving session: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    """Reset a session to its initial state."""
    try:
        # Create a new session with the same ID, effectively overwriting the old one
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Delete existing session
                cur.execute("DELETE FROM sessions WHERE chat_name = %s", (session_id,))
                # Delete associated questions
                cur.execute("DELETE FROM questions WHERE chat_name = %s", (session_id,))
                # Delete associated examinations
                cur.execute("DELETE FROM examinations WHERE chat_name = %s", (session_id,))
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Database error: {e}")
        finally:
            conn.close()
        
        # Create a new session
        session = create_or_get_session(session_id)
        
        # Return the first question
        current_question = questions_init[0]
        
        # Format the output based on the question type
        if current_question['type'] in ['MC', 'MCM', 'YN']:
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in current_question['options']])
            output = f"{current_question['question']}\n\n{options_text}"
        else:
            output = current_question['question']
        
        return jsonify({
            "status": "success",
            "message": "Session reset successfully",
            "output": output,
            "metadata": {
                "phase": "initial",
                "question_type": current_question['type'],
                "options": current_question.get('options', []),
                "question_index": 0
            }
        })
    
    except Exception as e:
        print(f"Error resetting session: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Health Assistant API is running"
    })

if __name__ == "__main__":
    # Get the port from environment variable or use default
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 to allow external access
    app.run(host="0.0.0.0", port=port)