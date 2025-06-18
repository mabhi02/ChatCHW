from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
import sys
import os
import openai
import datetime
from pymongo import MongoClient
from chad import (
    questions_init,
    structured_questions_array,
    examination_history,
    get_embedding_batch,
    vectorQuotesWithSource,
    process_with_matrix,
    judge,
    parse_examination_text,
    get_diagnosis_and_treatment,
    parse_question_data,
    store_examination,
    initialize_session,
    get_session_data,
)
from prompts import MedicalPrompts

# Import MATRIXConfig for configuration constants
from AVM.MATRIX.config import MATRIXConfig

# Import properly for NeonDB - use psycopg2 for native connection
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    print("Successfully imported psycopg2")
except ImportError:
    print("Failed to import psycopg2, attempting fallback")
    try:
        from psycopg import connect
        psycopg2 = connect
        print("Using psycopg fallback")
    except ImportError:
        print("No PostgreSQL drivers available")
        psycopg2 = None

app = Flask(__name__)
# Set frontend URL with fallback to your deployed frontend
frontend_url = os.environ.get('FRONTEND_URL', 'https://chw-demo.onrender.com')
# Configure CORS to allow requests from both localhost and your deployed frontend
CORS(app, origins=["http://localhost:3000", "https://localhost:3000", frontend_url], 
     supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Global state management
sessions = {}

# Initialize NeonDB with the correct URL
neondb_url = os.environ.get('DATABASE_URL') or os.environ.get('NEON_DATABASE_URL')
db_conn = None
if psycopg2 and neondb_url:
    try:
        # Connect to the database
        db_conn = psycopg2.connect(neondb_url)
        db_conn.autocommit = True  # Use autocommit mode
        
        # Test connection with a simple query
        with db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                print("NeonDB connected successfully")
            else:
                print("NeonDB connection test failed")
                db_conn = None
    except Exception as e:
        print(f"Error connecting to NeonDB: {e}")
        db_conn = None
else:
    print("NeonDB configuration not available")

def setup_database_tables():
    """Set up necessary database tables if they don't exist"""
    if not db_conn:
        print("Database connection not available, skipping table setup")
        return
    
    try:
        with db_conn.cursor() as cursor:
            # Create rag_chunks table if it doesn't exist
            create_chunks_table = """
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                phase TEXT,
                chunk_type TEXT,
                chunk_id TEXT,
                source TEXT,
                text TEXT,
                relevance_score FLOAT
            )
            """
            cursor.execute(create_chunks_table)
            
            # Create session_summaries table
            create_summaries_table = """
            CREATE TABLE IF NOT EXISTS session_summaries (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                initial_complaint TEXT,
                diagnosis TEXT,
                treatment TEXT,
                chunks_used INTEGER
            )
            """
            cursor.execute(create_summaries_table)
            
            # Create matrix_evaluations table
            create_matrix_table = """
            CREATE TABLE IF NOT EXISTS matrix_evaluations (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                phase TEXT,
                question TEXT,
                confidence FLOAT,
                optimist_weight FLOAT,
                pessimist_weight FLOAT,
                selected_agent TEXT
            )
            """
            cursor.execute(create_matrix_table)
            
            # Create conversation_messages table to store complete conversation history
            create_messages_table = """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                message_type TEXT NOT NULL,
                phase TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB
            )
            """
            cursor.execute(create_messages_table)
            
            print("Database tables setup complete")
    except Exception as e:
        print(f"Error setting up database tables: {e}")

# Run table setup at startup
setup_database_tables()

def save_session_chunks_to_neondb(session_id, phase, chunks, chunk_type="retrieved"):
    """Save chunks from RAG process to NeonDB for logging and analysis"""
    if not db_conn:
        print("NeonDB not available, skipping chunk logging")
        return
    
    try:
        # Get timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        with db_conn.cursor() as cursor:
            # Log each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.get('id', f'unknown-{i}')
                source = chunk.get('source', 'unknown')
                text = chunk.get('text', '')
                score = float(chunk.get('score', 0.0))
                
                # Insert into NeonDB
                insert_query = """
                INSERT INTO rag_chunks (
                    session_id, timestamp, phase, chunk_type, chunk_id, source, text, relevance_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (
                    session_id, 
                    timestamp, 
                    phase, 
                    chunk_type, 
                    chunk_id, 
                    source, 
                    text, 
                    score
                ))
        
        print(f"Saved {len(chunks)} chunks to NeonDB for session {session_id}")
    except Exception as e:
        print(f"Error saving chunks to NeonDB: {e}")

def save_matrix_evaluation(session_id, phase, question, matrix_output):
    """Save MATRIX evaluation results to NeonDB"""
    if not db_conn:
        print("NeonDB not available, skipping MATRIX evaluation logging")
        return
    
    try:
        # Get timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        with db_conn.cursor() as cursor:
            # Insert MATRIX evaluation
            insert_query = """
            INSERT INTO matrix_evaluations (
                session_id, timestamp, phase, question, confidence, 
                optimist_weight, pessimist_weight, selected_agent
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                session_id,
                timestamp,
                phase,
                question,
                matrix_output.get('confidence', 0.0),
                matrix_output.get('weights', {}).get('optimist', 0.5),
                matrix_output.get('weights', {}).get('pessimist', 0.5),
                matrix_output.get('selected_agent', 'unknown')
            ))
        
        print(f"Saved MATRIX evaluation for question '{question}' in session {session_id}")
    except Exception as e:
        print(f"Error saving MATRIX evaluation: {e}")

def save_conversation_message(session_id, message_type, phase, content, metadata=None):
    """Save a conversation message to the database
    
    Args:
        session_id (str): The session ID
        message_type (str): 'user' or 'assistant'
        phase (str): 'initial', 'followup', 'exam', or 'complete'
        content (str): The message content
        metadata (dict): Optional metadata about the message
    """
    if not db_conn:
        print("NeonDB not available, skipping conversation message logging")
        return
    
    try:
        # Get timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        with db_conn.cursor() as cursor:
            # Convert metadata to JSON string if provided
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            # Insert message
            insert_query = """
            INSERT INTO conversation_messages (
                session_id, timestamp, message_type, phase, content, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """
            cursor.execute(insert_query, (
                session_id,
                timestamp,
                message_type,
                phase,
                content,
                metadata_json
            ))
        
        print(f"Saved {message_type} message to conversation history for session {session_id}")
    except Exception as e:
        print(f"Error saving conversation message to NeonDB: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ChatCHW Medical Assessment API",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new assessment session"""
    try:
        data = request.get_json()
        if not data:
            data = {}
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required"
            }), 400

        # Initialize session
        session_data = get_session_data(session_id, sessions)
        
        # Store session_id in the session data
        session_data['session_id'] = session_id
        
        # Save user message for session start
        save_conversation_message(
            session_id=session_id,
            message_type='system',
            phase='initial',
            content='Assessment session started'
        )
        
        # Get first question from the initial questions array
        first_question = questions_init[0]
        question_text = first_question['question']
        
        if first_question['type'] in ['MC', 'MCM']:
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
            output = f"{question_text}\n\n{options_text}"
        else:
            output = question_text
        
        # Save assistant message for first question
        save_conversation_message(
            session_id=session_id,
            message_type='assistant',
            phase='initial',
            content=output,
            metadata={
                'question_type': first_question['type'],
                'options': first_question.get('options', [])
            }
        )
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "initial",
                "question_type": first_question['type'],
                "options": first_question.get('options', [])
            }
        })
        
    except Exception as e:
        print(f"Error in start_assessment: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to start assessment: {str(e)}"
        }), 500

@app.route('/api/input', methods=['POST'])
def process_input():
    """Process user input and return next question/response"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        user_input = data.get('input')
        
        print(f"Received input: {user_input} for session: {session_id}")  # Debug print
        
        session_data = get_session_data(session_id, sessions)
        print(f"Current phase: {session_data['phase']}")  # Debug print
        
        # Save user message to conversation history
        save_conversation_message(
            session_id=session_id,
            message_type='user',
            phase=session_data['phase'],
            content=user_input
        )
        
        if session_data['phase'] == "initial":
            try:
                # Store initial response
                current_question = questions_init[session_data['current_question_index']]
                session_data['initial_responses'].append({
                    "question": current_question['question'],
                    "answer": user_input,
                    "type": current_question['type']
                })
                
                print(f"Stored response for question {session_data['current_question_index']}")  # Debug print
                
                # Move to next question or phase
                session_data['current_question_index'] += 1
                if session_data['current_question_index'] < len(questions_init):
                    next_question = questions_init[session_data['current_question_index']]
                    if next_question['type'] in ['MC', 'MCM']:
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                        output = f"{next_question['question']}\n\n{options_text}"
                    else:
                        output = next_question['question']
                        
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='initial',
                        content=output,
                        metadata={
                            'question_type': next_question['type'],
                            'options': next_question.get('options', [])
                        }
                    )
                        
                    return jsonify({
                        "status": "success",
                        "output": output,
                        "metadata": {
                            "phase": session_data['phase'],
                            "question_type": next_question['type'],
                            "options": next_question.get('options', [])
                        }
                    })
                else:
                    print("Moving to followup phase")  # Debug print
                    session_data['phase'] = "followup"
                    response = generate_followup_question(session_data)
                    
                    # Save assistant message to conversation history if phase changes
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='followup',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in initial phase: {str(e)}")  # Debug print
                raise
                
        elif session_data['phase'] == "followup":
            try:
                # Store followup response
                if 'current_followup_question' in session_data:
                    session_data['followup_responses'].append({
                        "question": session_data['current_followup_question']['question'],
                        "answer": user_input,
                        "type": "MC"
                    })
                
                # Generate next followup question or move to exam phase
                if judge(session_data['followup_responses'], session_data['current_followup_question']['question']):
                    session_data['phase'] = "exam"
                    response = generate_examination(session_data)
                    
                    # Save assistant message to conversation history for phase change
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='exam',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                else:
                    response = generate_followup_question(session_data)
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='followup',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in followup phase: {str(e)}")  # Debug print
                raise
                
        elif session_data['phase'] == "exam":
            try:
                # Store examination response
                if 'current_examination' in session_data:
                    # Fix: Match the text to the option ID
                    selected_option = 1  # Default to first option
                    
                    # Try to find which option was selected by matching the text
                    if 'options' in session_data['current_examination']:
                        for option in session_data['current_examination']['options']:
                            if option['text'] == user_input:
                                selected_option = option['id']
                                break
                    
                    # Check if this is a NO_INFO_EXAM type (no examination available)
                    if session_data['current_examination'].get('type') == "NO_INFO_EXAM":
                        # If user selected "Proceed with general assessment and diagnosis"
                        if selected_option == 1 or "proceed with general assessment" in user_input.lower():
                            # Add a generic response to exam_responses so we can proceed
                            session_data['exam_responses'].append({
                                "examination": "No specific examination was available in the medical guide.",
                                "result": "Proceeded with general assessment based on symptoms and history",
                                "type": "INFORMAL_GUIDANCE"
                            })
                            # Go directly to diagnosis phase
                            session_data['phase'] = "complete"
                            response = generate_final_results(session_data)
                            
                            # Save assistant message for diagnosis
                            if response.status_code == 200:
                                response_data = response.get_json()
                                save_conversation_message(
                                    session_id=session_id,
                                    message_type='assistant',
                                    phase='complete',
                                    content=response_data['output'],
                                    metadata=response_data['metadata']
                                )
                            
                            return response
                        elif selected_option == 2 or "refer to higher" in user_input.lower():
                            # Add a referral response
                            session_data['exam_responses'].append({
                                "examination": "No specific examination was available in the medical guide.",
                                "result": "Patient referred to higher level facility",
                                "type": "REFERRAL"
                            })
                            # Still proceed to diagnosis for documentation
                            session_data['phase'] = "complete" 
                            response = generate_final_results(session_data)
                            
                            # Save assistant message for diagnosis with referral
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='complete',
                                content=response.json['output'],
                                metadata={**response.json['metadata'], 'referral': True}
                            )
                            
                            return response
                    
                    # For regular examination types, continue with original logic
                    examination_text = session_data['current_examination']['text']
                    
                    # Determine if this was a formal examination or not based on content
                    examination, _ = parse_examination_text(examination_text)
                    is_formal_exam = not any(phrase in examination.lower() for phrase in [
                        "does not provide", "no information", "no examination", 
                        "no specific examination", "doesn't provide"
                    ]) and not examination.startswith("PARTIAL INFORMATION")
                    
                    # Store examination regardless of type for session tracking
                    store_examination(examination_text, selected_option)
                    
                    # Add response but mark clearly if it wasn't a formal examination
                    if is_formal_exam:
                        exam_type = "FORMAL_EXAM"
                    else:
                        exam_type = "INFORMAL_GUIDANCE"
                        
                    session_data['exam_responses'].append({
                        "examination": examination,
                        "result": user_input,
                        "type": exam_type
                    })
                
                # Generate next examination or complete assessment
                # Check if the current examination was a NO_INFO_EXAM type
                has_no_info_exam = any(resp.get('type') == 'INFORMAL_GUIDANCE' and 
                                       ("does not provide" in resp.get('examination', '').lower() or 
                                        "no specific examination" in resp.get('examination', '').lower() or 
                                        "no examination" in resp.get('examination', '').lower())
                                      for resp in session_data['exam_responses'])
                
                # If we've had at least one formal examination or we've tried 3 times, proceed to diagnosis
                formal_exam_count = sum(1 for resp in session_data['exam_responses'] if resp.get('type') == 'FORMAL_EXAM')
                total_exam_count = len(session_data['exam_responses'])
                
                if has_no_info_exam:
                    # If there's no examination info, skip MATRIX and proceed to diagnosis
                    print("\nNo examination info in medical guide, proceeding directly to diagnosis.")
                    session_data['phase'] = "complete"
                    response = generate_final_results(session_data)
                    
                    # Save assistant message for diagnosis
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='complete',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                elif (formal_exam_count > 0 and judge_exam(session_data['exam_responses'], session_data['current_examination']['text'])) or total_exam_count >= 3:
                    session_data['phase'] = "complete"
                    response = generate_final_results(session_data)
                    
                    # Save assistant message for diagnosis
                    if response.status_code == 200:
                        response_data = response.get_json()
                        save_conversation_message(
                            session_id=session_id,
                            message_type='assistant',
                            phase='complete',
                            content=response_data['output'],
                            metadata=response_data['metadata']
                        )
                    
                    return response
                else:
                    response = generate_examination(session_data)
                    
                    # Save assistant message for next examination
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='exam',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in exam phase: {str(e)}")  # Debug print
                raise
                
    except Exception as e:
        print(f"Error in process_input: {str(e)}")  # Debug print
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_followup_question(session_data):
    """Generate and format followup question for chat interface"""
    try:
        initial_responses = session_data.get('initial_responses', [])
        followup_responses = session_data.get('followup_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        if not initial_complaint:
            return jsonify({
                "status": "error",
                "message": "No initial complaint found"
            })
        
        # Get relevant medical guide content
        try:
            embedding = get_embedding_batch([initial_complaint])[0]
            relevant_docs = vectorQuotesWithSource(embedding, pc.Index("who-guide-old"))
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            relevant_docs = []
        
        if not relevant_docs:
            # Fallback question if no relevant docs
            question = "Can you describe any additional symptoms you are experiencing?"
            options = [
                {"id": 1, "text": "Pain or discomfort"},
                {"id": 2, "text": "Fever or chills"},
                {"id": 3, "text": "Difficulty breathing"},
                {"id": 4, "text": "Nausea or vomiting"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        else:
            # Generate question based on medical guide content
            medical_content = "\n".join([doc["text"] for doc in relevant_docs[:2]])
            
            try:
                prompt = f"""Based on the medical guide content and the patient's complaint of "{initial_complaint}", what is the most important follow-up question to ask?

Medical guide content:
{medical_content}

Generate a single, specific follow-up question that would help determine the appropriate care."""

                response = openai.ChatCompletion.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": "You are a medical assistant helping with patient assessment. Generate clear, specific follow-up questions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.3
                )
                question = response.choices[0].message.content.strip()
                
                # Generate simple options
                options = [
                    {"id": 1, "text": "Yes"},
                    {"id": 2, "text": "No"},
                    {"id": 3, "text": "Sometimes"},
                    {"id": 4, "text": "Not sure"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            except Exception as e:
                print(f"Error generating question with OpenAI: {e}")
                question = "Can you describe any additional symptoms you are experiencing?"
                options = [
                    {"id": 1, "text": "Pain or discomfort"},
                    {"id": 2, "text": "Fever or chills"},
                    {"id": 3, "text": "Difficulty breathing"},
                    {"id": 4, "text": "Nausea or vomiting"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
        
        # Store current question in session
        session_data['current_followup_question'] = {
            "question": question,
            "options": options,
            "type": "MC"
        }
        
        return jsonify({
            "status": "success",
            "output": question,
            "metadata": {
                "phase": "followup",
                "question_type": "MC",
                "options": options
            }
        })
        
    except Exception as e:
        print(f"Error generating followup question: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })
     
def generate_examination(session_data):
    """Generate examination recommendation using session data"""
    try:
        initial_responses = session_data.get('initial_responses', [])
        followup_responses = session_data.get('followup_responses', [])
        exam_responses = session_data.get('exam_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        if not initial_complaint:
            return jsonify({
                "status": "error",
                "message": "No initial complaint found"
            })
        
        # Get relevant medical guide content
        try:
            embedding = get_embedding_batch([initial_complaint])[0]
            relevant_docs = vectorQuotesWithSource(embedding, pc.Index("who-guide-old"))
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            relevant_docs = []
        
        if not relevant_docs:
            return jsonify({
                "status": "success",
                "output": "Based on the symptoms described, the medical guide does not provide specific examination procedures for this condition. Please proceed with general assessment and refer to a health facility if needed.",
                "metadata": {
                    "phase": "exam",
                    "question_type": "NO_INFO_EXAM",
                    "type": "NO_INFO_EXAM",
                    "options": [
                        {"id": 1, "text": "Proceed with general assessment and diagnosis"},
                        {"id": 2, "text": "Refer to health facility"}
                    ]
                }
            })
        
        # Generate examination using medical guide content
        medical_content = "\n".join([doc["text"] for doc in relevant_docs[:2]])
        
        try:
            prompt = f"""Based on the medical guide content and patient complaint "{initial_complaint}", what examination should be performed?

Medical guide content:
{medical_content}

Provide a specific examination recommendation with clear steps."""

            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "system", "content": "You are a medical assistant providing examination guidance based on WHO medical guidelines."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            examination_text = response.choices[0].message.content.strip()
            
            # Store current examination in session
            session_data['current_examination'] = {
                "text": examination_text,
                "type": "FORMAL_EXAM",
                "options": [
                    {"id": 1, "text": "Normal findings"},
                    {"id": 2, "text": "Abnormal findings detected"},
                    {"id": 3, "text": "Unable to perform examination"},
                    {"id": 4, "text": "Other findings"}
                ]
            }
            
            return jsonify({
                "status": "success",
                "output": examination_text,
                "metadata": {
                    "phase": "exam",
                    "question_type": "MC",
                    "options": session_data['current_examination']['options']
                }
            })
            
        except Exception as e:
            print(f"Error generating examination: {e}")
            return jsonify({
                "status": "success",
                "output": "Please perform a general physical examination based on the patient's symptoms and refer to a health facility if needed.",
                "metadata": {
                    "phase": "exam",
                    "question_type": "MC",
                    "options": [
                        {"id": 1, "text": "Normal findings"},
                        {"id": 2, "text": "Abnormal findings detected"},
                        {"id": 3, "text": "Unable to perform examination"},
                        {"id": 4, "text": "Other findings"}
                    ]
                }
            })
        
    except Exception as e:
        print(f"Error in generate_examination: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_final_results(session_data):
    """Generate and format final results using session data"""
    try:
        initial_responses = session_data.get('initial_responses', [])
        followup_responses = session_data.get('followup_responses', [])
        exam_responses = session_data.get('exam_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        if not initial_complaint:
            return jsonify({
                "status": "error",
                "message": "No initial complaint found for diagnosis"
            })
        
        # Use the existing diagnosis function from chad.py
        try:
            results = get_diagnosis_and_treatment(
                initial_responses,
                followup_responses,
                exam_responses
            )
            
            diagnosis_text = results.get("diagnosis", "Unable to determine diagnosis from available information")
            treatment_text = results.get("treatment", "General care recommended - refer to health facility")
            
            output = f"""Assessment Complete

Diagnosis:
{diagnosis_text}

Treatment Plan:
{treatment_text}

Note: This assessment is based on the WHO medical guide for community health workers. For complex cases or if symptoms persist, refer to a health facility.
"""
            
            return jsonify({
                "status": "success",
                "output": output,
                "metadata": {
                    "phase": "complete"
                }
            })
            
        except Exception as e:
            print(f"Error getting diagnosis: {e}")
            return jsonify({
                "status": "success",
                "output": f"""Assessment Complete

Based on the complaint: {initial_complaint}

Recommendation: Please refer the patient to a health facility for proper evaluation and treatment as the specific condition requires professional medical assessment.

Note: This assessment is based on available information. Professional medical evaluation is recommended.
""",
                "metadata": {
                    "phase": "complete"
                }
            })
        
    except Exception as e:
        print(f"Error in generate_final_results: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> bool:
    """
    Judge examination similarity with improved duplicate detection.
    
    OVERRIDE: Always return True for NO_INFO cases to proceed to diagnosis.
    """
    # Check if this is a "no information" case, if so, always proceed to diagnosis
    if ("does not provide" in current_exam.lower() or 
        "no specific examination" in current_exam.lower() or 
        "the guide doesn't provide" in current_exam.lower() or
        "no information" in current_exam.lower()):
        print("\nNo examination information in guide, proceeding to diagnosis.")
        return True
    
    if not previous_exams:
        return False
        
    try:
        if len(previous_exams) >= MATRIXConfig.MAX_EXAMS:
            print("\nReached maximum number of examinations.")
            return True
            
        exam_lines = current_exam.split('\n')
        current_exam_name = ""
        current_procedure = ""
        
        for line in exam_lines:
            if line.startswith("Examination:"):
                current_exam_name = line.split('Examination:')[1].strip().lower()
            elif line.startswith("Procedure:"):
                current_procedure = line.split('Procedure:')[1].strip().lower()
        
        for exam in previous_exams:
            prev_exam_lines = exam['examination'].split('\n')
            prev_name = ""
            prev_procedure = ""
            
            for line in prev_exam_lines:
                if line.startswith("Examination:"):
                    prev_name = line.split('Examination:')[1].strip().lower()
                elif line.startswith("Procedure:"):
                    prev_procedure = line.split('Procedure:')[1].strip().lower()
            
            if (prev_name in current_exam_name or current_exam_name in prev_name):
                print(f"\nSimilar examination '{current_exam_name}' has already been performed.")
                return True
                
            if len(prev_procedure) > 0 and len(current_procedure) > 0:
                words1 = set(prev_procedure.split())
                words2 = set(current_procedure.split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if similarity > 0.7:
                    print(f"\nVery similar procedure has already been performed.")
                    return True
        
        matrix_output = process_with_matrix(
            current_exam, 
            previous_exams
        )
        
        should_end = (
            matrix_output["confidence"] > MATRIXConfig.EXAM_SIMILARITY_THRESHOLD or
            len(previous_exams) >= MATRIXConfig.MAX_EXAMS or
            matrix_output["weights"]["optimist"] > 0.8
        )
        
        if should_end:
            print("\nSufficient examinations completed based on comprehensive analysis.")
            
        return should_end
                
    except Exception as e:
        print(f"Warning: Exam similarity check falling back to basic method: {e}")
        return len(previous_exams) >= MATRIXConfig.MAX_EXAMS

@app.route('/api/conversation/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get the conversation history for a session"""
    try:
        if not db_conn:
            return jsonify({
                "status": "error",
                "message": "Database connection not available"
            })
        
        with db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get all messages for this session, ordered by timestamp
            query = """
            SELECT * FROM conversation_messages 
            WHERE session_id = %s 
            ORDER BY timestamp ASC
            """
            cursor.execute(query, (session_id,))
            messages = cursor.fetchall()
            
            # Convert PostgreSQL types to JSON-serializable types
            for message in messages:
                # Convert timestamp to string
                if 'timestamp' in message and message['timestamp']:
                    message['timestamp'] = message['timestamp'].isoformat()
            
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "message_count": len(messages),
                "messages": messages
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == "__main__":
    # Get the port from Render's environment variable
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 to allow external access
    app.run(host="0.0.0.0", port=port)
