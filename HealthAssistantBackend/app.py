from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
import sys
import os
import openai
import datetime
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
frontend_url = os.environ.get('FRONTEND_URL', 'https://chatchw.onrender.com')
# Configure CORS to allow requests from both localhost and your deployed frontend
CORS(app, origins=["http://localhost:3000", frontend_url], supports_credentials=True)
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

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new session and return first question"""
    try:
        session_id = request.json.get('session_id', 'default')
        session_data = get_session_data(session_id, sessions)
        
        # Store session_id in the session data
        session_data['session_id'] = session_id
        
        # Format first question for chat interface
        first_question = questions_init[0]
        question_text = first_question['question']
        
        if first_question['type'] in ['MC', 'MCM']:
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
            output = f"{question_text}\n\n{options_text}"
        else:
            output = question_text
            
        # Save assistant message to conversation history
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
        return jsonify({
            "status": "error",
            "message": str(e)
        })
        
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
    """Generate and format followup question"""
    try:
        print("Starting generate_followup_question")  # Debug print
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        session_id = session_data.get('session_id', 'default')
        print(f"Initial complaint: {initial_complaint}")  # Debug print
        
        # Extract previously gathered information to avoid repeating questions
        patient_info = {}
        
        # Get patient sex
        sex_response = next((resp for resp in session_data['initial_responses'] 
                        if "sex" in resp['question'].lower()), None)
        if sex_response:
            patient_info['sex'] = sex_response['answer']
            
        # Get patient age
        age_response = next((resp for resp in session_data['initial_responses'] 
                        if "age" in resp['question'].lower()), None)
        if age_response:
            patient_info['age'] = age_response['answer']
            
        # Get patient symptoms from initial complaint and followup responses
        patient_info['symptoms'] = [initial_complaint]
        for resp in session_data['followup_responses']:
            patient_info['symptoms'].append(f"Q: {resp['question']} - A: {resp['answer']}")
        
        # Build context including the patient information we've already collected
        context = f"Initial complaint: {initial_complaint}\n"
        context += f"Patient profile: {patient_info}\n"
        if session_data['followup_responses']:
            context += "Previous responses:\n"
            for resp in session_data['followup_responses']:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        # Get embeddings and relevant documents
        index = pc.Index("who-guide")
        embedding = get_embedding_batch([context])[0]
        
        print("Got embedding")  # Debug print
        
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)  # Increased to get more context
        
        print(f"Got {len(relevant_docs)} relevant docs")  # Debug print
        print("Relevant docs:")  # Debug print
        print("" + "-"*50)  # Debug print
        for doc in relevant_docs:   
            print(f"Doc: {doc['text']} (score: {doc['score']})")    # Debug print 
            print("" + "+"*50)  # Debug print
        print("-"*50)  # Debug print
        
        if not relevant_docs:
            raise Exception("Could not generate relevant question")
        
        # Store chunks for reference
        if 'retrieved_chunks' not in session_data:
            session_data['retrieved_chunks'] = []
        session_data['retrieved_chunks'].extend(relevant_docs)
        
        # Save chunks to NeonDB for persistence
        try:
            save_session_chunks_to_neondb(
                session_id=session_id,
                phase="followup_question",
                chunks=relevant_docs,
                chunk_type="retrieved"
            )
        except Exception as e:
            print(f"Error saving followup chunks to NeonDB: {e}")
        
        # Format medical guide information
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs[:3]])
        
        # Create list of previous questions to avoid repetition
        previous_questions = [resp['question'] for resp in session_data['followup_responses']]
        previous_questions_text = "\n".join([f"- {q}" for q in previous_questions])
        
        # Create list of information we already know to avoid asking about it again
        known_info = []
        if 'sex' in patient_info:
            known_info.append(f"- Patient sex: {patient_info['sex']}")
        if 'age' in patient_info:
            known_info.append(f"- Patient age: {patient_info['age']}")
        known_info_text = "\n".join(known_info) if known_info else "- No additional information"
        
        prompt = f'''Patient information:
Initial complaint: "{initial_complaint}"

Information we already know (DO NOT ASK ABOUT THESE AGAIN):
{known_info_text}

Previous questions already asked (DO NOT REPEAT THESE):
{previous_questions_text if previous_questions else "- No previous follow-up questions"}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. The question must be based directly on assessment questions mentioned in the medical guide.
4. If the medical guide mentions specific questions to ask for this condition, use those exact questions.
5. DO NOT ask about information we already know (like age or sex).
6. DO NOT repeat any previous questions that have already been asked.
7. Focus on gathering NEW information that is not already known.
8. Return only the question text without any explanation.

Based ONLY on the medical guide information, what follow-up question should be asked to assess this patient?
'''
        
        print("Sending prompt to OpenAI")  # Debug print

        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        question = response.choices[0].message.content.strip()

        print(f"Got question: {question}")  # Debug print
        
        # MATRIX evaluation - analyze the question's relevance given previous questions
        if previous_questions:
            # Process the question through MATRIX system for pattern recognition
            matrix_output = process_with_matrix(question, session_data['followup_responses'], context)
            print(f"MATRIX evaluation - confidence: {matrix_output.get('confidence', 0)}, selected agent: {matrix_output.get('selected_agent', 'unknown')}")
            
            # Save MATRIX evaluation to database
            try:
                save_matrix_evaluation(
                    session_id=session_id,
                    phase="followup_question",
                    question=question,
                    matrix_output=matrix_output
                )
            except Exception as e:
                print(f"Error saving MATRIX evaluation: {e}")
            
            # Check if the question is too similar to previous ones
            similarity_threshold = 0.7  # Configurable threshold
            if matrix_output.get('confidence', 0) > similarity_threshold:
                print(f"Question '{question}' is too similar to previous questions (confidence: {matrix_output.get('confidence', 0)})")
                
                # Check if we have enough information to proceed to examination phase
                if len(session_data['followup_responses']) >= 3:
                    print("Sufficient information gathered. Moving to examination phase.")
                    session_data['phase'] = "exam"
                    return generate_examination(session_data)
        else:
            # For the first question, create a default matrix output
            matrix_output = {
                "confidence": 0.1,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.9, "pessimist": 0.1}
            }
        
        # Generate options
        options_prompt = f'''QUESTION: "{question}"

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. Generate 4 answer options for the above question.
4. If the medical guide mentions specific answer options, use those exactly.
5. If not, create relevant options based only on the medical guide content.
6. Format as a numbered list (1-4).
7. Do not include explanations, only the options themselves.
'''
        
        options_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
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
        
        options.append({"id": 5, "text": "Other (please specify)"})
        
        print(f"Generated {len(options)} options")  # Debug print
        
        # Store the generated question in session
        session_data['current_followup_question'] = {
            "question": question,
            "options": options,
            "sources": relevant_docs[:3],  # Store sources used for this question
            "matrix_output": matrix_output  # Store MATRIX evaluation output
        }
        
        # Format output for chat interface
        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in options])
        output = f"{question}\n\n{options_text}"
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "followup",
                "question_type": "MC",
                "options": options,
                "sources": [{"source": doc["source"], "score": doc["score"]} for doc in relevant_docs[:3]],
                "matrix_evaluation": {
                    "confidence": matrix_output.get('confidence', 0),
                    "selected_agent": matrix_output.get('selected_agent', 'unknown'),
                    "optimist_weight": matrix_output.get('weights', {}).get('optimist', 0.5),
                    "pessimist_weight": matrix_output.get('weights', {}).get('pessimist', 0.5)
                }
            }
        })
        
    except Exception as e:
        print(f"Error in generate_followup_question: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        return jsonify({
            "status": "error",
            "message": str(e)
        })
     
def generate_examination(session_data):
    """Generate examination using embeddings for context matching"""
    try:
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        session_id = session_data.get('session_id', 'default')
        
        # Extract previously gathered information to avoid redundant examinations
        patient_info = {}
        
        # Get patient sex
        sex_response = next((resp for resp in session_data['initial_responses'] 
                        if "sex" in resp['question'].lower()), None)
        if sex_response:
            patient_info['sex'] = sex_response['answer']
            
        # Get patient age
        age_response = next((resp for resp in session_data['initial_responses'] 
                        if "age" in resp['question'].lower()), None)
        if age_response:
            patient_info['age'] = age_response['answer']
        
        # Get previous examination information
        previous_exams = []
        if 'exam_responses' in session_data and session_data['exam_responses']:
            for exam in session_data['exam_responses']:
                if 'examination' in exam:
                    previous_exams.append(exam['examination'])
        
        # Get patient symptoms
        symptoms = []
        symptoms.append(initial_complaint)
        for resp in session_data['followup_responses']:
            answer = resp['answer'].lower()
            if any(symptom in answer for symptom in 
                  ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                   'headache', 'nausea', 'dizziness', 'rash']):
                symptoms.append(answer)
        
        # Build comprehensive context with all patient info
        context_items = [initial_complaint]
        context_items.append(f"Patient: {patient_info}")
        context_items.extend(symptoms)
        
        # Build a complete context string for MATRIX evaluation
        context_text = f"Initial complaint: {initial_complaint}\n" + \
                       f"Patient info: {patient_info}\n" + \
                       f"Symptoms: {', '.join(symptoms)}\n" + \
                       f"Previous exams: {', '.join(previous_exams)}"
        
        # Get embeddings for all context items
        embeddings = get_embedding_batch(context_items)
        
        # Get relevant documents using embeddings
        index = pc.Index("who-guide")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=2)
            if matches:
                relevant_matches.extend(matches)
        print("Here are the chunks that got matched")
        print(relevant_matches)  # Debug print
        
        # Store chunks for reference
        if 'examination_chunks' not in session_data:
            session_data['examination_chunks'] = []
        session_data['examination_chunks'].extend(relevant_matches)
        
        # Save chunks to NeonDB for persistence
        try:
            save_session_chunks_to_neondb(
                session_id=session_id,
                phase="examination",
                chunks=relevant_matches,
                chunk_type="retrieved"
            )
        except Exception as e:
            print(f"Error saving examination chunks to NeonDB: {e}")
        
        # Sort matches by relevance score
        relevant_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top matches, ensure we don't have duplicates
        seen_ids = set()
        unique_matches = []
        for match in relevant_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                unique_matches.append(match)
        
        top_matches = unique_matches[:3]  # Take up to 3 unique matches
        
        # Format context with just the text content
        medical_guide_content = "\n\n".join([match['text'] for match in top_matches])
        
        # Format information on previous exams to avoid repeating them
        previous_exams_text = "\n".join([f"- {exam}" for exam in previous_exams]) if previous_exams else "- No previous examinations"
        
        # Create list of information we already know to inform examination choice
        known_info = []
        if 'sex' in patient_info:
            known_info.append(f"- Patient sex: {patient_info['sex']}")
        if 'age' in patient_info:
            known_info.append(f"- Patient age: {patient_info['age']}")
        # Add symptom information
        for i, symptom in enumerate(symptoms[:5]):  # Limit to 5 key symptoms
            known_info.append(f"- Symptom {i+1}: {symptom}")
        known_info_text = "\n".join(known_info) if known_info else "- No additional information"
        
        # Build a prompt focused only on the medical guide content
        symptoms_summary = ", ".join(symptoms[:3]) if symptoms else "No specific symptoms identified"
        
        prompt = f'''Patient information:
Initial complaint: "{initial_complaint}"
Key symptoms: {symptoms_summary}

Information we already know:
{known_info_text}

Previous examinations already performed (DO NOT REPEAT THESE):
{previous_exams_text}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. You MUST select one of these three formats for your response:

FORMAT 1 - WHEN THE GUIDE CLEARLY PROVIDES AN EXAMINATION PROCEDURE:
[Examination name]
[Detailed procedure steps as described in the guide]
#: [Finding 1]
#: [Finding 2]
#: [Finding 3]
#: [Finding 4]

FORMAT 2 - WHEN THE GUIDE PROVIDES NO INFORMATION ABOUT EXAMINATIONS:
"The medical guide does not provide specific examination procedures for this condition."

FORMAT 3 - WHEN THE GUIDE PROVIDES PARTIAL/INFORMAL INFORMATION BUT NO CLEAR PROCEDURE:
"While the medical guide does not provide a formal examination procedure, here is some relevant information that may be helpful:"
[Include any relevant assessment guidance from the guide]

4. Use the same terminology and procedures as presented in the medical guide.
5. DO NOT repeat any examinations that have already been performed.
6. Choose an examination that is DIFFERENT from previous ones and appropriate for the patient's symptoms.

Based ONLY on the medical guide information, what NEW examination should be performed?
'''

        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        examination_text = completion.choices[0].message.content.strip()
        print(f"Generated examination text: {examination_text}")  # Debug print
        
        # Check for simple "no information" responses directly
        if examination_text == "The guide doesn't provide that information." or examination_text == "The medical guide does not provide specific examination procedures for this condition.":
            # Generate simplified options for no examination info
            options = [
                {"id": 1, "text": "Proceed with general assessment and diagnosis"},
                {"id": 2, "text": "Refer to higher level facility"},
                {"id": 3, "text": "Other (please specify)"}
            ]
            
            # Save this as a special type of "NO_INFO" examination
            session_data['current_examination'] = {
                "text": examination_text,
                "options": options,
                "sources": top_matches,
                "matrix_output": {
                    "confidence": 0.1,
                    "selected_agent": "optimist",
                    "weights": {"optimist": 0.9, "pessimist": 0.1}
                },
                "type": "NO_INFO_EXAM"
            }
            
            return jsonify({
                "status": "success",
                "output": f"The medical guide does not provide specific examination procedures for this condition.\n\nPlease select how you would like to proceed:",
                "metadata": {
                    "phase": "exam",
                    "question_type": "EXAM",
                    "options": options,
                    "sources": [{"source": doc["source"], "score": doc["score"]} for doc in top_matches],
                    "matrix_evaluation": {
                        "confidence": 0.1,
                        "selected_agent": "optimist",
                        "optimist_weight": 0.9,
                        "pessimist_weight": 0.1
                    }
                }
            })
        
        # Use MATRIX to evaluate examination relevance and similarity
        if session_data['exam_responses']:
            try:
                matrix_output = process_with_matrix(examination_text, session_data['exam_responses'], context_text)
                print(f"MATRIX evaluation - confidence: {matrix_output.get('confidence', 0)}, selected agent: {matrix_output.get('selected_agent', 'unknown')}")
                
                # Save MATRIX evaluation to database
                try:
                    save_matrix_evaluation(
                        session_id=session_id,
                        phase="examination",
                        question=examination_text.split('\n')[0][:50],  # Use first line (exam name) as identifier
                        matrix_output=matrix_output
                    )
                except Exception as e:
                    print(f"Error saving MATRIX evaluation: {e}")
                    
                # If we've reached the confidence threshold for similarity, we might need to generate more diverse examinations
                similarity_threshold = 0.7
                if matrix_output.get('confidence', 0) > similarity_threshold:
                    print(f"Warning: Generated examination is too similar to previous ones (confidence: {matrix_output.get('confidence', 0)})")
                    
                    # If this is consistent across multiple attempts, we might want to change approach
                    if len(session_data['exam_responses']) >= 3 and matrix_output.get('weights', {}).get('optimist', 0) > 0.7:
                        print("MATRIX suggests moving to diagnosis phase")
                        session_data['phase'] = "complete"
                        return generate_final_results(session_data)
            except Exception as e:
                print(f"Error in MATRIX evaluation: {e}")
                matrix_output = {
                    "confidence": 0.1,
                    "selected_agent": "optimist",
                    "weights": {"optimist": 0.9, "pessimist": 0.1}
                }
        else:
            # For the first examination, create a default matrix output
            matrix_output = {
                "confidence": 0.1,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.9, "pessimist": 0.1}
            }
        
        # Parse the examination text - this will now handle cases where the guide doesn't provide examination info
        try:
            examination, option_texts = parse_examination_text(examination_text)
            
            options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts[:4])]
            options.append({"id": 5, "text": "Other (please specify)"})
            
            session_data['current_examination'] = {
                "text": examination_text,
                "options": options,
                "sources": top_matches,  # Store sources used for this examination
                "matrix_output": matrix_output  # Store MATRIX evaluation
            }
            
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in options])
            
            # Determine which scenario we're in based on the examination text
            if any(phrase in examination.lower() for phrase in ["does not provide", "no information", "no examination", 
                                                            "no specific examination", "doesn't provide"]):
                # SCENARIO 1: No examination information available
                output = f"Examination Information:\n{examination}\n\nRecommended Action:\n{options_text}"
                
                # Add a note about source material
                if top_matches:
                    source_info = "\n\nNote: This recommendation is based on the WHO medical guide which does not provide specific examination procedures for this condition."
                    output += source_info
                    
            elif examination.startswith("PARTIAL INFORMATION"):
                # SCENARIO 2: Partial/informal examination information
                # Extract the partial information without the prefix
                partial_info = examination.replace("PARTIAL INFORMATION (Not a formal examination procedure):\n", "")
                
                output = f"Note: The medical guide does not provide a formal examination procedure for this condition.\n\n"
                output += f"Relevant Information:\n{partial_info}\n\n"
                output += f"Recommended Action:\n{options_text}"
                
                # Add citation to show where information came from
                if top_matches:
                    source_names = ", ".join([match.get('source', 'Medical guide').split('/')[-1] for match in top_matches[:2]])
                    output += f"\n\nSource: {source_names}"
            else:
                # SCENARIO 3: Clear examination with findings
                output = f"Recommended Examination:\n{examination}\n\nFindings:\n{options_text}"
            
            return jsonify({
                "status": "success",
                "output": output,
                "metadata": {
                    "phase": "exam",
                    "question_type": "EXAM",
                    "options": options,
                    "sources": [{"source": doc["source"], "score": doc["score"]} for doc in top_matches],
                    "matrix_evaluation": {
                        "confidence": matrix_output.get('confidence', 0),
                        "selected_agent": matrix_output.get('selected_agent', 'unknown'),
                        "optimist_weight": matrix_output.get('weights', {}).get('optimist', 0.5),
                        "pessimist_weight": matrix_output.get('weights', {}).get('pessimist', 0.5)
                    }
                }
            })
            
        except Exception as e:
            print(f"Error parsing examination text: {e}")
            # Fallback options if parsing fails
            options = [
                {"id": 1, "text": "Proceed with general assessment and diagnosis"},
                {"id": 2, "text": "Refer to higher level facility"},
                {"id": 3, "text": "Other (please specify)"}
            ]
            
            # Save this as a special type of "NO_INFO" examination
            session_data['current_examination'] = {
                "text": "The system encountered an issue while determining an appropriate examination based on the medical guide.",
                "options": options,
                "sources": [],
                "matrix_output": {
                    "confidence": 0.1,
                    "selected_agent": "optimist",
                    "weights": {"optimist": 0.9, "pessimist": 0.1}
                },
                "type": "NO_INFO_EXAM"
            }
            
            return jsonify({
                "status": "success",
                "output": "The system encountered an issue while determining an appropriate examination based on the medical guide.\n\nPlease select how you would like to proceed:",
                "metadata": {
                    "phase": "exam",
                    "question_type": "EXAM",
                    "options": options,
                    "sources": [],
                    "matrix_evaluation": {
                        "confidence": 0.1,
                        "selected_agent": "unknown",
                        "optimist_weight": 0.5,
                        "pessimist_weight": 0.5
                    }
                }
            })
    except Exception as e:
        print(f"Error in generate_examination: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback response if an error occurs
        options = [
            {"id": 1, "text": "Proceed with general assessment and diagnosis"},
            {"id": 2, "text": "Refer to higher level facility"},
            {"id": 3, "text": "Other (please specify)"}
        ]
        
        # Save this as a special type of "NO_INFO" examination
        session_data['current_examination'] = {
            "text": "The system encountered an issue while determining an appropriate examination based on the medical guide.",
            "options": options,
            "sources": [],
            "matrix_output": {
                "confidence": 0.1,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.9, "pessimist": 0.1}
            },
            "type": "NO_INFO_EXAM"
        }
        
        return jsonify({
            "status": "success",
            "output": "The system encountered an issue while determining an appropriate examination based on the medical guide.\n\nPlease select how you would like to proceed:",
            "metadata": {
                "phase": "exam",
                "question_type": "EXAM",
                "options": options,
                "sources": [],
                "matrix_evaluation": {
                    "confidence": 0.1,
                    "selected_agent": "unknown",
                    "optimist_weight": 0.5,
                    "pessimist_weight": 0.5
                }
            }
        })

def generate_final_results(session_data):
    """Generate and format final results"""
    try:
        session_id = session_data.get('session_id', 'default')
        
        # Check if we proceeded without proper examination
        has_no_formal_exam = all(resp.get('type') != 'FORMAL_EXAM' for resp in session_data['exam_responses'])
        was_referred = any(resp.get('type') == 'REFERRAL' for resp in session_data['exam_responses'])
        
        # Add special context for diagnosis when no formal examination was done
        if has_no_formal_exam:
            initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                if resp['question'] == "Please describe what brings you here today"), "")
            
            # Extract symptoms from followup responses
            symptoms = []
            for resp in session_data['followup_responses']:
                if isinstance(resp.get('answer'), str) and not resp.get('answer').startswith("No "):
                    symptoms.append(resp.get('answer'))
            
            # If patient was referred, create a referral summary
            if was_referred:
                # Create a simple referral summary
                return jsonify({
                    "status": "success",
                    "output": f"""Assessment Summary
                    
Initial complaint: {initial_complaint}

Key findings:
{chr(10).join([f"- {symptom}" for symptom in symptoms[:5]])}

Diagnosis:
The medical guide does not provide specific examination procedures for this condition. 
A full diagnosis requires proper medical examination.

Recommendation:
Patient has been referred to a higher level facility for proper examination and diagnosis.

Note: This summary is based on limited information and should not be considered a complete medical assessment.
""",
                    "metadata": {
                        "phase": "complete",
                        "referral": True,
                        "formal_examination": False,
                        "citations": []
                    }
                })
        
        # Get diagnosis and treatment based on responses
        results = get_diagnosis_and_treatment(
            session_data['initial_responses'],
            session_data['followup_responses'],
            session_data['exam_responses']
        )
        
        # Store chunks used for reference if available
        if 'chunks_used' in results:
            session_data['chunks_used'] = results['chunks_used']
            
            # Save chunks to NeonDB for persistence
            try:
                save_session_chunks_to_neondb(
                    session_id=session_id,
                    phase="diagnosis_treatment",
                    chunks=results['chunks_used'],
                    chunk_type="used_for_generation"
                )
            except Exception as e:
                print(f"Error saving diagnosis chunks to NeonDB: {e}")
        
        # Format results for chat interface, preserving formatting
        diagnosis_text = results['diagnosis']
        treatment_text = results['treatment']
        
        # Format citations
        citations_text = ""
        if results['citations']:
            citations = []
            for cite in results['citations']:
                citations.append(f"- {cite['source']} (relevance: {cite['score']:.2f})")
            citations_text = "\n".join(citations)
        else:
            citations_text = "No specific medical guide citations available."
        
        # Add a note about limitations of assessment if no formal examination was done
        note = ""
        if has_no_formal_exam:
            note = "\nNote: This assessment is based on limited information without formal examination procedures described in the medical guide. Consider referring for a complete medical evaluation if symptoms persist or worsen."
        
        output = f"""Assessment Complete

Diagnosis:
{diagnosis_text}

Treatment Plan:
{treatment_text}
{note}

Key References:
{citations_text}
"""
        
        # Also save a summary record of the entire session
        try:
            if db_conn:
                # Create session_summaries table if it doesn't exist
                create_table_query = """
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
                with db_conn.cursor() as cursor:
                    cursor.execute(create_table_query)
                    
                    # Insert summary record
                    initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                    if resp['question'] == "Please describe what brings you here today"), "")
                    
                    chunks_count = len(results.get('chunks_used', []))
                    
                    insert_query = """
                    INSERT INTO session_summaries (
                        session_id, timestamp, initial_complaint, diagnosis, treatment, chunks_used
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        session_id,
                        datetime.datetime.now().isoformat(),
                        initial_complaint,
                        diagnosis_text,
                        treatment_text,
                        chunks_count
                    ))
                db_conn.commit()
                print(f"Saved session summary to NeonDB for session {session_id}")
        except Exception as e:
            print(f"Error saving session summary to NeonDB: {e}")
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "complete",
                "citations": results['citations'],
                "chunks_used": results.get('chunks_used', []),
                "formal_examination": not has_no_formal_exam
            }
        })
        
    except Exception as e:
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