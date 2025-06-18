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

# MongoDB setup
mongodb_url = os.environ.get('MONGODB_URL', 'mongodb://localhost:27017/')
mongodb_client = MongoClient(mongodb_url)
db = mongodb_client.medical_assessment
sessions_collection = db.sessions

def create_or_get_session(chat_name: str) -> Dict[str, Any]:
    """Create a new session or get existing session from MongoDB."""
    session = sessions_collection.find_one({"chat_name": chat_name})
    if not session:
        # Initialize new session
        session = {
            "chat_name": chat_name,
            "initial_responses": [],
            "followup_responses": [],
            "exam_responses": [],
            "current_question_index": 0,
            "phase": "initial",
            "created_at": datetime.datetime.now().isoformat()
        }
        sessions_collection.insert_one(session)
    return session

def update_session_responses(chat_name: str, response_type: str, response_data: Dict[str, Any]):
    """Update session responses in MongoDB."""
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$push": {response_type: response_data}}
    )

def update_session_phase(chat_name: str, new_phase: str):
    """Update session phase in MongoDB."""
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$set": {"phase": new_phase}}
    )

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
                    response = generate_examination(session_data['session_id'])
                    
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
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='complete',
                                content=response.json['output'],
                                metadata=response.json['metadata']
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
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='complete',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                else:
                    response = generate_examination(session_data['session_id'])
                    
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
    """Generate and format followup question using MongoDB session"""
    try:
        # Get session data
        session = create_or_get_session(session_data['session_id'])
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get patient info for context
        patient_info = {
            'age': next((resp['answer'] for resp in initial_responses 
                        if resp['question'] == "What is the patient's age?"), None),
            'sex': next((resp['answer'] for resp in initial_responses 
                        if resp['question'] == "What is the patient's sex?"), None)
        }
        
        # Format known information
        known_info = []
        if patient_info['sex']:
            known_info.append(f"- Patient sex: {patient_info['sex']}")
        if patient_info['age']:
            known_info.append(f"- Patient age: {patient_info['age']}")
        known_info_text = "\n".join(known_info) if known_info else "- No additional information"
        
        # Format previous questions
        previous_questions = [resp['question'] for resp in followup_responses]
        previous_questions_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else ""
        
        # Get relevant medical guide content
        embedding = get_embedding_batch([initial_complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, pc.Index("who-guide-old"))
        
        if not relevant_docs:
            print("No relevant medical guide content found")
            return jsonify({
                "status": "error",
                "message": "Could not find relevant medical information"
            })
            
        medical_guide_content = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
        
        # Generate question using advanced prompt
        prompt = MedicalPrompts.get_advanced_followup_prompt(
            initial_complaint=initial_complaint,
            known_info_text=known_info_text,
            previous_questions_text=previous_questions_text,
            medical_guide_content=medical_guide_content
        )
        
        print("Sending prompt to OpenAI")
        
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": MedicalPrompts.SYSTEM_MESSAGES['followup_question']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        question = response.choices[0].message.content.strip()
        
        print(f"Got question: {question}")
        
        # Generate options using advanced prompt
        options_prompt = MedicalPrompts.get_advanced_options_prompt(
            question=question,
            medical_guide_content=medical_guide_content
        )
        
        options_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": MedicalPrompts.SYSTEM_MESSAGES['options']},
                {"role": "user", "content": options_prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        options = []
        for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
            if opt.strip():
                text = opt.strip()
                if text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                options.append({"id": i+1, "text": text})
        
        # Add "Other" option
        options.append({"id": len(options) + 1, "text": "Other (please specify)"})
        
        # Store current question in session
        current_followup = {
            "question": question,
            "options": options,
            "type": "MC"
        }
        sessions_collection.update_one(
            {"chat_name": session_data['session_id']},
            {"$set": {"current_followup_question": current_followup}}
        )
        
        # Save relevant chunks to NeonDB
        save_session_chunks_to_neondb(session_data['session_id'], "followup", relevant_docs)
        
        return jsonify({
            "status": "success",
            "output": question,
            "metadata": {
                "phase": "followup",
                "question_type": "MC",
                "options": options,
                "citations": relevant_docs[-5:]
            }
        })
        
    except Exception as e:
        print(f"Error generating followup question: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })
     
def generate_examination(chat_name):
    """Generate examination recommendation using MongoDB session"""
    try:
        # Get session data
        session = create_or_get_session(chat_name)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        exam_responses = session.get('exam_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Format symptoms summary
        symptoms = []
        for resp in followup_responses:
            symptoms.append(f"- {resp['question']}: {resp['answer']}")
        symptoms_summary = "\n".join(symptoms) if symptoms else "No additional symptoms reported"
        
        # Format known information
        known_info = []
        for resp in initial_responses:
            if resp['question'] != "Please describe what brings you here today":
                known_info.append(f"- {resp['question']}: {resp['answer']}")
        for resp in followup_responses:
            known_info.append(f"- {resp['question']}: {resp['answer']}")
        known_info_text = "\n".join(known_info) if known_info else "No additional information"
        
        # Format previous examinations
        previous_exams = []
        for exam in exam_responses:
            previous_exams.append(f"- {exam['examination']}: {exam['result']}")
        previous_exams_text = "\n".join(previous_exams) if previous_exams else ""
        
        # Get relevant medical guide content
        embedding = get_embedding_batch([initial_complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, pc.Index("who-guide-old"))
        
        if not relevant_docs:
            print("No relevant medical guide content found")
            return jsonify({
                "status": "error",
                "message": "Could not find relevant medical information"
            })
            
        medical_guide_content = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
        
        # Generate examination using advanced prompt
        prompt = MedicalPrompts.get_advanced_examination_prompt(
            initial_complaint=initial_complaint,
            symptoms_summary=symptoms_summary,
            known_info_text=known_info_text,
            previous_exams_text=previous_exams_text,
            medical_guide_content=medical_guide_content
        )
        
        print("Sending prompt to OpenAI")
        
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": MedicalPrompts.SYSTEM_MESSAGES['examination']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        examination_text = response.choices[0].message.content.strip()
        print(f"Got examination: {examination_text}")
        
        # Check if no examination information is available
        if "does not provide" in examination_text.lower():
            return jsonify({
                "status": "success",
                "output": examination_text,
                "metadata": {
                    "phase": "exam",
                    "question_type": "MC",
                    "options": MedicalPrompts.NO_INFO_RESPONSES['proceed_options'],
                    "citations": relevant_docs[-5:]
                }
            })
            
        # Parse examination text into structured format
        examination, findings = parse_examination_text(examination_text)
        
        # Create options from findings
        options = [{"id": i+1, "text": finding} for i, finding in enumerate(findings)]
        
        # Store current examination in session
        current_exam = {
            "examination": examination,
            "options": options,
            "type": "MC"
        }
        sessions_collection.update_one(
            {"chat_name": chat_name},
            {"$set": {"current_examination": current_exam}}
        )
        
        # Save relevant chunks to NeonDB
        save_session_chunks_to_neondb(chat_name, "examination", relevant_docs)
        
        return jsonify({
            "status": "success",
            "output": examination,
            "metadata": {
                "phase": "exam",
                "question_type": "MC",
                "options": options,
                "citations": relevant_docs[-5:]
            }
        })
        
    except Exception as e:
        print(f"Error generating examination: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_final_results(chat_name):
    """Generate final diagnosis and treatment recommendations using MongoDB session"""
    try:
        # Get session data
        session = create_or_get_session(chat_name)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        exam_responses = session.get('exam_responses', [])
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get diagnosis and treatment recommendations
        results = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
        
        if 'error' in results:
            return jsonify({
                "status": "error",
                "message": results['error']
            })
            
        # Format the assessment summary
        if not exam_responses:
            # If no examinations were performed, use referral summary template
            summary = MedicalPrompts.REFERRAL_SUMMARY_TEMPLATE.format(
                initial_complaint=initial_complaint,
                key_findings="\n".join([f"- {resp['question']}: {resp['answer']}" 
                                      for resp in followup_responses])
            )
        else:
            # Format complete assessment with all findings
            diagnosis_text = results['diagnosis']
            treatment_text = "\n".join([
                f"Immediate Care:\n{results['immediate_care']}",
                f"\nHome Care:\n{results['home_care']}",
                f"\nReferral Guidance:\n{results['referral']}"
            ])
            
            # Add limitation note if no formal examinations were described
            limitation_note = ""
            if any("does not provide" in exam['examination'].lower() for exam in exam_responses):
                limitation_note = MedicalPrompts.LIMITATION_NOTES['no_formal_exam']
            
            # Format citations from chunks used
            citations = []
            if 'retrieved_chunks' in session:
                citations = [f"- {chunk['source']}" for chunk in session['retrieved_chunks'][-5:]]
            citations_text = "\n".join(citations) if citations else "No specific citations available"
            
            summary = MedicalPrompts.COMPLETE_ASSESSMENT_TEMPLATE.format(
                diagnosis_text=diagnosis_text,
                treatment_text=treatment_text,
                limitation_note=limitation_note,
                citations_text=citations_text
            )
        
        # Save summary to NeonDB
        try:
            with db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO session_summaries (
                        session_id, timestamp, initial_complaint, 
                        diagnosis, treatment, chunks_used
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    chat_name,
                    datetime.datetime.now().isoformat(),
                    initial_complaint,
                    results.get('diagnosis', ''),
                    treatment_text if 'treatment_text' in locals() else '',
                    len(session.get('retrieved_chunks', []))
                ))
        except Exception as e:
            print(f"Error saving summary to NeonDB: {e}")
        
        return jsonify({
            "status": "success",
            "output": summary,
            "metadata": {
                "phase": "complete",
                "diagnosis": results.get('diagnosis', ''),
                "immediate_care": results.get('immediate_care', ''),
                "home_care": results.get('home_care', ''),
                "referral": results.get('referral', '')
            }
        })
        
    except Exception as e:
        print(f"Error generating final results: {str(e)}")
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
