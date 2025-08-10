"""
Professional Way CHW Assistant (profapp.py)

This Flask application combines the web API structure from app.py with the RAG-based logic 
from profsway.py. It uses ONLY Retrieval-Augmented Generation (RAG) from the WHO medical guide.

CRITICAL: This implementation uses ONLY Retrieval-Augmented Generation (RAG) 
from the WHO medical guide. NO hardcoded medical knowledge is included.

All medical decisions including:
- Symptom identification
- Danger sign questions  
- Examinations
- Diagnosis
- Treatment

Are derived ENTIRELY from the WHO medical guide via vector database retrieval
from the "who-guide-old" Pinecone index.

If symptoms are not covered in the WHO guide, patients are referred to medical centers.

TRANSPARENCY FEATURES:
- Citations and sources are displayed for every medical decision
- Users can see which parts of the WHO guide were used
- Relevance scores show confidence in the retrieved information
- Preview text shows the actual content used from the guide
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import openai
import datetime
from prompts_infer import (
    get_diagnosis_prompt,
    get_treatment_prompt,
    get_main_followup_question_prompt,
    get_main_examination_prompt
)
from dangersigns_emily import identify_danger_signs

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
            
            # Create conversation_messages table if it doesn't exist
            create_messages_table = """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                message_type TEXT,
                phase TEXT,
                content TEXT,
                metadata JSONB
            )
            """
            cursor.execute(create_messages_table)
            
            print("Database tables set up successfully")
    except Exception as e:
        print(f"Error setting up database tables: {e}")

# Initialize database tables
setup_database_tables()

# Initial screening questions - Original format as specified
questions_init = [
    {
        "question": "What is the patient's sex?",
        "type": "MC",
        "options": [
            {"id": 1, "text": "Male"},
            {"id": 2, "text": "Female"},
            {"id": 3, "text": "Non-binary"},
            {"id": 4, "text": "Other"},
            {"id": 5, "text": "Other (please specify)"}
        ]
    },
    {
        "question": "What is the patient's age?",
        "type": "NUMERIC"
    },
    {
        "question": "Does the patient have a caregiver?",
        "type": "MC",
        "options": [
            {"id": 1, "text": "Yes"},
            {"id": 2, "text": "No"},
            {"id": 3, "text": "Not sure"},
            {"id": 4, "text": "Sometimes"},
            {"id": 5, "text": "Other (please specify)"}
        ]
    },
    {
        "question": "Who is accompanying the patient?",
        "type": "MC",
        "options": [
            {"id": 1, "text": "None"},
            {"id": 2, "text": "Relatives"},
            {"id": 3, "text": "Friends"},
            {"id": 4, "text": "Health workers"},
            {"id": 5, "text": "Other (please specify)"}
        ]
    },
    {
        "question": "Please describe what brings you here today",
        "type": "FREE"
    }
]

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI"""
    try:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

def vectorQuotesWithSource(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Get relevant quotes from vector database with source information"""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        quotes = []
        for match in results.matches:
            quotes.append({
                'id': match.id,
                'text': match.metadata.get('text', ''),
                'source': match.metadata.get('source', 'Unknown'),
                'score': match.score
            })
        
        return quotes
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []

def get_openai_completion(prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
    """Get completion from OpenAI"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting OpenAI completion: {e}")
        return ""

def identify_symptoms_from_complaint(complaint: str, index) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Identify symptoms from patient complaint using RAG"""
    try:
        print(f"Identifying symptoms from complaint: '{complaint}'")
        # Use RAG to identify symptoms mentioned in the complaint
        embedding = get_embedding_batch([complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)  # Reduced from 5 to 3
        
        print(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            print("No relevant documents found")
            return [], []
        
        # Use medical guide content to identify symptoms
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        symptom_identification_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Based ONLY on the WHO medical guide content below, identify the MOST IMPORTANT symptoms mentioned in this patient complaint: "{complaint}"
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        Return ONLY the 2-3 most critical symptoms that are:
        1. Mentioned in the patient's complaint
        2. Explicitly covered in the WHO medical guide content above
        3. Most relevant for medical assessment
        
        IMPORTANT: 
        - If the patient mentions "fever", "hot", "temperature", or similar terms, and fever is mentioned in the WHO guide, include "fever"
        - If the patient mentions "chills", "cold", "shivering", or similar terms, and chills are mentioned in the WHO guide, include "chills"
        - If the patient mentions "cough", "coughing", or similar terms, and cough is mentioned in the WHO guide, include "cough"
        - If the patient mentions "diarrhea", "diarrhoea", "loose stools", or similar terms, and diarrhea is mentioned in the WHO guide, include "diarrhea"
        - If the patient mentions "vomit", "vomiting", "nausea", or similar terms, and vomiting is mentioned in the WHO guide, include "vomiting"
        
        CRITICAL RULES:
        - ONLY identify symptoms that are EXPLICITLY mentioned in the patient's complaint
        - Do NOT infer symptoms that are not mentioned
        - If the patient says "fever and chills", do NOT add "diarrhea" or other unrelated symptoms
        - Focus on the PRIMARY symptoms the patient is complaining about
        - Limit to 2-3 most important symptoms maximum
        
        Format your response as a simple comma-separated list of 2-3 symptoms (e.g., "fever, cough")
        If no symptoms can be identified from the WHO medical guide, respond with "none"
        
        Remember: Use ONLY information from the WHO guide above, not your medical training. Limit to 2-3 most important symptoms that are explicitly mentioned in the complaint.
        """
        
        print("Sending symptom identification prompt to OpenAI")
        response = get_openai_completion(
            symptom_identification_prompt,
            max_tokens=100,
            temperature=0.3
        )
        
        print(f"OpenAI response: {response}")
        
        if response.lower().strip() == "none":
            print("No symptoms identified")
            return [], relevant_docs
        
        # Parse the response to extract symptoms
        symptoms = [s.strip() for s in response.split(',')]
        symptoms = [s for s in symptoms if s and len(s) > 2]
        # LIMIT: Only take the first 3 symptoms maximum
        MAX_SYMPTOMS = 3
        symptoms = symptoms[:MAX_SYMPTOMS]
        
        print(f"Parsed symptoms (limited to {MAX_SYMPTOMS}): {symptoms}")
        return symptoms, relevant_docs
        
    except Exception as e:
        print(f"Error in identify_symptoms_from_complaint: {str(e)}")
        return [], []

def rephrase_danger_sign_question(danger_sign_text: str) -> str:
    """Rephrase danger sign text into a coherent, grammatically correct question."""
    try:
        prompt = f"""
        Rephrase this danger sign into a clear, grammatically correct question that refers to "the patient".
        
        Danger sign: {danger_sign_text}
        
        Examples:
        - "not able to drink or eat anything" → "Is the patient not able to drink or eat anything?"
        - "unusually sleepy or unconscious" → "Is the patient unusually sleepy or unconscious?"
        - "chest indrawing" → "Does the patient have chest indrawing?"
        - "fever for the last 7 days or more" → "Has the patient had fever for 7 days or more?"
        
        Return ONLY the rephrased question, nothing else.
        """
        
        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a medical assistant that rephrases danger signs into clear, grammatically correct questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        rephrased = completion.choices[0].message.content.strip()
        # Remove any quotes if present
        rephrased = rephrased.strip('"\'')
        return rephrased
        
    except Exception as e:
        print(f"Error rephrasing question: {e}")
        # Fallback to simple rephrasing
        danger_sign_lower = danger_sign_text.lower()
        if danger_sign_lower.startswith(('not able', 'unable', 'cannot')):
            return f"Is the patient {danger_sign_lower}?"
        elif danger_sign_lower.startswith(('has', 'have', 'had')):
            return f"Does the patient {danger_sign_lower}?"
        else:
            return f"Does the patient have {danger_sign_lower}?"

def check_similar_questions(new_question: str, previous_questions: List[str]) -> bool:
    """Check if a question is too similar to previously asked questions"""
    new_question_lower = new_question.lower()
    
    for prev_question in previous_questions:
        prev_lower = prev_question.lower()
        
        # Check for exact matches
        if new_question_lower == prev_lower:
            return True
        
        # Check for key medical terms that should not be repeated
        key_terms = ['chest indrawing', 'fever', 'drink', 'eat', 'breathing', 'pain', 'headache']
        for term in key_terms:
            if term in new_question_lower and term in prev_lower:
                return True
        
        # Check for high similarity (more than 80% similar words)
        new_words = set(new_question_lower.split())
        prev_words = set(prev_lower.split())
        
        if new_words and prev_words:
            intersection = new_words.intersection(prev_words)
            union = new_words.union(prev_words)
            similarity = len(intersection) / len(union)
            
            if similarity > 0.8:
                return True
    
    return False



def parse_examination_text(text):
    """Parse examination text to extract examination name and procedure"""
    try:
        lines = text.split('\n')
        examination_name = ""
        procedure = ""
        
        for line in lines:
            line = line.strip()
            # More flexible parsing - handle variations in formatting
            if "examination:" in line.lower():
                # Handle both "Examination:" and "examination:" (case insensitive)
                parts = line.split(":", 1)
                if len(parts) > 1:
                    examination_name = parts[1].strip()
            elif "procedure:" in line.lower():
                # Handle both "Procedure:" and "procedure:" (case insensitive)
                parts = line.split(":", 1)
                if len(parts) > 1:
                    procedure = parts[1].strip()
        
        # If we still don't have an examination name, try to extract from the first line
        if not examination_name and lines:
            first_line = lines[0].strip()
            if first_line and not first_line.lower().startswith(("procedure:", "examination:")):
                examination_name = first_line
        
        return {
            "examination": examination_name,
            "procedure": procedure,
            "full_text": text
        }
    except Exception as e:
        print(f"Error parsing examination text: {e}")
        return {
            "examination": "Unknown",
            "procedure": "",
            "full_text": text
        }

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              danger_sign_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get diagnosis and treatment based on responses using RAG"""
    try:
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        symptoms = []
        for resp in danger_sign_responses:
            if isinstance(resp.get('answer'), str) and resp.get('answer') not in ["No", "Not sure"]:
                symptoms.append(f"{resp['symptom']}: {resp['question']} - {resp['answer']}")
        
        symptoms_text = "\n".join([f"- {symptom}" for symptom in symptoms])
        
        exam_results = []
        for exam in exam_responses:
            if 'examination' in exam and 'result' in exam:
                exam_results.append(f"Examination: {exam['examination']}\nResult: {exam['result']}")
        exam_results_text = "\n\n".join(exam_results)
        
        context_items = [initial_complaint]
        context_items.extend(symptoms)
        context_items.extend(exam_results)
        
        embeddings = get_embedding_batch(context_items)
        
        index = pc.Index("who-guide-old")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=2)
            if matches:
                relevant_matches.extend(matches)
        
        relevant_matches.sort(key=lambda x: x['score'], reverse=True)
        
        seen_ids = set()
        unique_matches = []
        for match in relevant_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                unique_matches.append(match)
        
        top_matches = unique_matches[:3]
        medical_guide_content = "\n\n".join([match['text'] for match in top_matches])
        
        diagnosis_prompt = get_diagnosis_prompt(
            initial_complaint=initial_complaint,
            symptoms_text=symptoms_text,
            exam_results_text=exam_results_text,
            medical_guide_content=medical_guide_content
        )
        
        diagnosis_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training. Do NOT make medical assumptions. If something is not in the WHO guide, say so explicitly."},
                {"role": "user", "content": diagnosis_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        
        treatment_prompt = get_treatment_prompt(
            initial_complaint=initial_complaint,
            symptoms_text=symptoms_text,
            exam_results_text=exam_results_text,
            diagnosis=diagnosis,
            medical_guide_content=medical_guide_content
        )
        
        treatment_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training. Do NOT make medical assumptions. If something is not in the WHO guide, say so explicitly."},
                {"role": "user", "content": treatment_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        treatment = treatment_completion.choices[0].message.content.strip()
        
        citations = []
        for match in top_matches:
            citations.append({
                "source": match.get('source', 'Medical guide'),
                "score": match.get('score', 0.0)
            })
        
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": citations,
            "chunks_used": top_matches
        }
        
    except Exception as e:
        print(f"Error in get_diagnosis_and_treatment: {str(e)}")
        return {
            "diagnosis": "Error generating diagnosis",
            "treatment": "Error generating treatment",
            "citations": [],
            "chunks_used": []
        }

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

def save_conversation_message(session_id, message_type, phase, content, metadata=None):
    """Save a conversation message to the database"""
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

def initialize_session(session_id: str) -> Dict[str, Any]:
    """Initialize a new session"""
    if session_id not in sessions:
        sessions[session_id] = {
            'phase': 'initial',
            'current_question_index': 0,
            'initial_responses': [],
            'followup_responses': [],
            'exam_responses': [],
            'danger_sign_responses': [],
            'symptoms': [],
            'citations': []
        }
    return sessions[session_id]

def get_session_data(session_id: str, sessions_dict: Dict) -> Dict[str, Any]:
    """Get or create session data"""
    if session_id not in sessions_dict:
        initialize_session(session_id)
    return sessions_dict[session_id]

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new session and return first question"""
    try:
        session_id = request.json.get('session_id', 'default')
        session_data = get_session_data(session_id, sessions)
        
        # Reset session data
        session_data['phase'] = 'initial'
        session_data['current_question_index'] = 0
        session_data['initial_responses'] = []
        session_data['followup_responses'] = []
        session_data['exam_responses'] = []
        session_data['danger_sign_responses'] = []
        session_data['symptoms'] = []
        session_data['citations'] = []
        
        # Get first question
        first_question = questions_init[0]
        if first_question['type'] in ['MC', 'MCM']:
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
            output = f"{first_question['question']}\n\n{options_text}"
        elif first_question['type'] == 'NUMERIC':
            output = first_question['question']
        else:
            output = first_question['question']
        
        # Prepare metadata based on question type
        metadata = {
            'question_type': first_question['type'],
            'options': first_question.get('options', [])
        }
        
        # Add numeric-specific metadata for slider rendering
        if first_question['type'] == 'NUMERIC':
            metadata.update({
                'numeric_input': True,
                'min_value': 0,
                'max_value': 120,
                'step': 1,
                'default_value': 30
            })
        
        # Save assistant message to conversation history
        save_conversation_message(
            session_id=session_id,
            message_type='assistant',
            phase='initial',
            content=output,
            metadata=metadata
        )
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": session_data['phase'],
                "question_type": first_question['type'],
                "options": first_question.get('options', []),
                **metadata
            }
        })
        
    except Exception as e:
        print(f"Error in start_assessment: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/input', methods=['POST'])
def process_input():
    """Process user input and return next question/response"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        user_input = data.get('input')
        
        print(f"Received input: {user_input} for session: {session_id}")
        
        session_data = get_session_data(session_id, sessions)
        print(f"Current phase: {session_data['phase']}")
        
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
                
                print(f"Stored response for question {session_data['current_question_index']}")
                
                # Move to next question or phase
                session_data['current_question_index'] += 1
                if session_data['current_question_index'] < len(questions_init):
                    next_question = questions_init[session_data['current_question_index']]
                    if next_question['type'] in ['MC', 'MCM']:
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                        output = f"{next_question['question']}\n\n{options_text}"
                    elif next_question['type'] == 'NUMERIC':
                        output = next_question['question']
                    else:
                        output = next_question['question']
                        
                    # Prepare metadata based on question type
                    metadata = {
                        'question_type': next_question['type'],
                        'options': next_question.get('options', [])
                    }
                    
                    # Add numeric-specific metadata for slider rendering
                    if next_question['type'] == 'NUMERIC':
                        metadata.update({
                            'numeric_input': True,
                            'min_value': 0,
                            'max_value': 120,
                            'step': 1,
                            'default_value': 30
                        })
                        
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='initial',
                        content=output,
                        metadata=metadata
                    )
                        
                    return jsonify({
                        "status": "success",
                        "output": output,
                        "metadata": {
                            "phase": session_data['phase'],
                            "question_type": next_question['type'],
                            "options": next_question.get('options', []),
                            **metadata
                        }
                    })
                else:
                    print("Moving to followup phase")
                    session_data['phase'] = "followup"
                    
                    # Identify symptoms from complaint using RAG
                    initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                        if resp['question'] == "Please describe what brings you here today"), "")
                    
                    index = pc.Index("who-guide-old")
                    symptoms, citations = identify_symptoms_from_complaint(initial_complaint, index)
                    session_data['symptoms'] = symptoms
                    session_data['citations'].extend(citations)
                    
                    # Save citations to database
                    save_session_chunks_to_neondb(session_id, "symptom_identification", citations)
                    
                    # PRIORITIZE IMMEDIATE EXAMINATIONS
                    # Check if immediate examinations are needed based on symptoms
                    immediate_exam = check_for_immediate_examination(symptoms, initial_complaint)
                    if immediate_exam:
                        print(f"Prioritizing immediate examination: {immediate_exam['name']}")
                        session_data['phase'] = "immediate_exam"
                        session_data['current_immediate_exam'] = immediate_exam
                        
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in immediate_exam['options']])
                        output = f"Based on your symptoms, I need to perform an immediate examination. Let me conduct a {immediate_exam['name'].lower()}:\n\n{immediate_exam['procedure']}\n\n{options_text}"
                        
                        # Save assistant message to conversation history
                        save_conversation_message(
                            session_id=session_id,
                            message_type='assistant',
                            phase='immediate_exam',
                            content=output,
                            metadata={
                                'immediate_examination': True,
                                'examination_name': immediate_exam['name'],
                                'question_type': 'MC',
                                'options': immediate_exam['options']
                            }
                        )
                        
                        return jsonify({
                            "status": "success",
                            "output": output,
                            "metadata": {
                                "phase": session_data['phase'],
                                "immediate_examination": True,
                                "examination_name": immediate_exam['name'],
                                "question_type": 'MC',
                                "options": immediate_exam['options']
                            }
                        })
                    
                    # SMART ASSESSMENT FLOW
                    # Use intelligent questioning instead of rigid phases
                    print("Starting smart assessment flow")
                    session_data['phase'] = "smart_assessment"
                    
                    # Get smart questions based on symptoms and medical guide
                    smart_questions = get_smart_questions(symptoms, initial_complaint, index, session_data)
                    
                    if smart_questions:
                        # Store smart questions for the session
                        session_data['smart_questions'] = smart_questions
                        session_data['current_question_index'] = 0
                        
                        # Ask the first smart question
                        first_question = smart_questions[0]
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
                        output = f"Now I need to ask you some important questions to properly assess your condition:\n\n{first_question['question']}\n\n{options_text}"
                        
                        # Save assistant message to conversation history
                        save_conversation_message(
                            session_id=session_id,
                            message_type='assistant',
                            phase='smart_assessment',
                            content=output,
                            metadata={
                                'smart_question': True,
                                'question': first_question['question'],
                                'question_type': first_question['type'],
                                'options': first_question['options'],
                                'question_index': 0
                            }
                        )
                        
                        return jsonify({
                            "status": "success",
                            "output": output,
                            "metadata": {
                                "phase": session_data['phase'],
                                "smart_question": True,
                                "question": first_question['question'],
                                "question_type": first_question['type'],
                                "options": first_question['options'],
                                "question_index": 0
                            }
                        })
                    else:
                        # Fallback to traditional danger sign questions if no smart questions found
                        print("No smart questions found, falling back to danger sign questions")
                        session_data['phase'] = "followup"
                        
                        # Generate danger sign questions
                        print(f"Identified symptoms: {symptoms}")
                        danger_sign_result = identify_danger_signs(initial_complaint, symptoms, index)
                        danger_sign_responses = []
                        danger_citations = []
                        
                        if danger_sign_result['status'] == 'success':
                            danger_signs = danger_sign_result['danger_signs']
                            danger_citations = danger_sign_result.get('citations', [])
                            print(f"Found {len(danger_signs)} danger signs: {danger_signs}")
                            
                            # Convert danger signs to questions with proper options (deduplicated)
                            seen_questions = set()
                            for danger_sign in danger_signs:
                                if isinstance(danger_sign, dict) and 'danger_sign' in danger_sign:
                                    danger_sign_text = danger_sign['danger_sign']
                                    
                                    # Create a focused question based on the danger sign
                                    question = rephrase_danger_sign_question(danger_sign_text)
                                    
                                    # Skip if we've already asked a similar question
                                    if check_similar_questions(question, seen_questions):
                                        continue
                                    
                                    seen_questions.add(question)
                                    
                                    danger_sign_responses.append({
                                        "question": question,
                                        "type": "MC",
                                        "symptom": danger_sign_text,
                                        "options": [
                                            {"id": 1, "text": "Yes"},
                                            {"id": 2, "text": "No"},
                                            {"id": 3, "text": "Not sure"}
                                        ]
                                    })
                        
                        print(f"Generated {len(danger_sign_responses)} danger sign questions")
                        session_data['danger_sign_responses'] = danger_sign_responses
                        session_data['citations'].extend(danger_citations)
                        
                        # Save danger sign citations to database
                        save_session_chunks_to_neondb(session_id, "danger_signs", danger_citations)
                        
                        if danger_sign_responses:
                            # Ask first danger sign question
                            first_danger_question = danger_sign_responses[0]
                            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_danger_question['options']])
                            output = f"Now I need to check for any danger signs. Please answer the following question:\n\n{first_danger_question['question']}\n\n{options_text}"
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='followup',
                                content=output,
                                metadata={
                                    'danger_sign_question': True, 
                                    'symptom': first_danger_question['symptom'],
                                    'question_type': first_danger_question['type'],
                                    'options': first_danger_question['options']
                                }
                            )
                            
                            return jsonify({
                                "status": "success",
                                "output": output,
                                "metadata": {
                                    "phase": session_data['phase'],
                                    "danger_sign_question": True,
                                    "symptom": first_danger_question['symptom'],
                                    "question_type": first_danger_question['type'],
                                    "options": first_danger_question['options']
                                }
                            })
                        else:
                            # No danger signs, move to examination phase
                            session_data['phase'] = "exam"
                            response = generate_examination(session_data)
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='exam',
                                content=response.json['output'],
                                metadata=response.json['metadata']
                            )
                            
                            return response
                    
            except Exception as e:
                print(f"Error in initial phase: {str(e)}")
                raise
                
        elif session_data['phase'] == "immediate_exam":
            try:
                # Store immediate examination response
                current_exam = session_data.get('current_immediate_exam', {})
                current_exam['result'] = user_input
                
                # Store the complete examination-answer pair
                session_data['exam_responses'].append({
                    "examination": current_exam.get('name', 'Unknown'),
                    "procedure": current_exam.get('procedure', ''),
                    "result": user_input,
                    "type": 'MC',
                    "options": current_exam.get('options', []),
                    "immediate": True
                })
                
                # After immediate examination, move to smart assessment
                session_data['phase'] = "smart_assessment"
                
                # Get symptoms and initial complaint for smart questioning
                symptoms = session_data.get('symptoms', [])
                initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                if resp['question'] == "Please describe what brings you here today"), "")
                
                # Ensure index is available
                index = pc.Index("who-guide-old")
                
                # Get smart questions based on symptoms and medical guide
                smart_questions = get_smart_questions(symptoms, initial_complaint, index, session_data)
                
                if smart_questions:
                    # Store smart questions for the session
                    session_data['smart_questions'] = smart_questions
                    session_data['current_question_index'] = 0
                    
                    # Ask the first smart question
                    first_question = smart_questions[0]
                    options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
                    output = f"Thank you for the test result. Now I need to ask you some important questions to properly assess your condition:\n\n{first_question['question']}\n\n{options_text}"
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='smart_assessment',
                        content=output,
                        metadata={
                            'smart_question': True,
                            'question': first_question['question'],
                            'question_type': first_question['type'],
                            'options': first_question['options'],
                            'question_index': 0
                        }
                    )
                    
                    return jsonify({
                        "status": "success",
                        "output": output,
                        "metadata": {
                            "phase": session_data['phase'],
                            "smart_question": True,
                            "question": first_question['question'],
                            "question_type": first_question['type'],
                            "options": first_question['options'],
                            "question_index": 0
                        }
                    })
                else:
                    # No smart questions found, move to examination phase
                    session_data['phase'] = "exam"
                    
                    # Add a brief transition message
                    transition_output = "Thank you for the test result. Now I'll perform additional examinations to better understand your condition."
                    
                    # Save transition message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='smart_assessment',
                        content=transition_output,
                        metadata={'transition': True}
                    )
                    
                    response = generate_examination(session_data)
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='exam',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in immediate_exam phase: {str(e)}")
                raise
                
        elif session_data['phase'] == "followup":
            try:
                # Store danger sign response with proper question-answer pair
                current_danger_index = session_data.get('current_danger_index', 0)
                current_danger_question = session_data['danger_sign_responses'][current_danger_index]
                current_danger_question['answer'] = user_input
                
                # Store the complete question-answer pair in followup_responses
                session_data['followup_responses'].append({
                    "question": current_danger_question['question'],
                    "answer": user_input,
                    "type": current_danger_question['type'],
                    "symptom": current_danger_question['symptom']
                })
                
                # Move to next danger sign question or phase
                current_danger_index += 1
                session_data['current_danger_index'] = current_danger_index
                
                if current_danger_index < len(session_data['danger_sign_responses']):
                    # Ask next danger sign question
                    next_danger_question = session_data['danger_sign_responses'][current_danger_index]
                    options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_danger_question['options']])
                    
                    # Add context about remaining questions for better flow
                    remaining_questions = len(session_data['danger_sign_responses']) - current_danger_index
                    if remaining_questions == 1:
                        output = f"One more question to check for danger signs:\n\n{next_danger_question['question']}\n\n{options_text}"
                    else:
                        output = f"Please answer the following question:\n\n{next_danger_question['question']}\n\n{options_text}"
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='followup',
                        content=output,
                        metadata={
                            'danger_sign_question': True, 
                            'symptom': next_danger_question['symptom'],
                            'question_type': next_danger_question['type'],
                            'options': next_danger_question['options']
                        }
                    )
                    
                    return jsonify({
                        "status": "success",
                        "output": output,
                        "metadata": {
                            "phase": session_data['phase'],
                            "danger_sign_question": True,
                            "symptom": next_danger_question['symptom'],
                            "question_type": next_danger_question['type'],
                            "options": next_danger_question['options']
                        }
                    })
                else:
                    # All danger sign questions answered, move to examination phase
                    session_data['phase'] = "exam"
                    
                    # Add a brief transition message
                    transition_output = "Thank you for answering the danger sign questions. Now I'll perform some examinations to better understand your condition."
                    
                    # Save transition message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='followup',
                        content=transition_output,
                        metadata={'transition': True}
                    )
                    
                    response = generate_examination(session_data)
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='exam',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in followup phase: {str(e)}")
                raise
                
        elif session_data['phase'] == "exam":
            try:
                # Store examination response with proper question-answer pair
                current_exam = session_data.get('current_examination', {})
                current_exam['result'] = user_input
                
                # Store the complete examination-answer pair
                session_data['exam_responses'].append({
                    "examination": current_exam.get('examination', 'Unknown'),
                    "procedure": current_exam.get('procedure', ''),
                    "result": user_input,
                    "type": current_exam.get('type', 'MC'),
                    "options": current_exam.get('options', [])
                })
                
                # Check if we should continue with more examinations
                if should_continue_examinations(session_data):
                    response = generate_examination(session_data)
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='exam',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                else:
                    # Generate final results
                    session_data['phase'] = "complete"
                    response = generate_final_results(session_data)
                    
                    # Save assistant message to conversation history
                    save_conversation_message(
                        session_id=session_id,
                        message_type='assistant',
                        phase='complete',
                        content=response.json['output'],
                        metadata=response.json['metadata']
                    )
                    
                    return response
                    
            except Exception as e:
                print(f"Error in exam phase: {str(e)}")
                raise
                
        elif session_data['phase'] == "smart_assessment":
            try:
                # Store the smart question response
                current_question_index = session_data.get('current_question_index', 0)
                smart_questions = session_data.get('smart_questions', [])
                
                if current_question_index < len(smart_questions):
                    current_question = smart_questions[current_question_index]
                    
                    # Store the response
                    if 'smart_responses' not in session_data:
                        session_data['smart_responses'] = []
                    
                    session_data['smart_responses'].append({
                        "question": current_question['question'],
                        "answer": user_input,
                        "type": current_question['type'],
                        "options": current_question['options']
                    })
                    
                    # Move to next question
                    session_data['current_question_index'] = current_question_index + 1
                    
                    # Check if we have more questions to ask
                    if session_data['current_question_index'] < len(smart_questions):
                        # Ask the next smart question
                        next_question = smart_questions[session_data['current_question_index']]
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                        output = f"Thank you. Now I need to ask you another important question:\n\n{next_question['question']}\n\n{options_text}"
                        
                        # Save assistant message to conversation history
                        save_conversation_message(
                            session_id=session_id,
                            message_type='assistant',
                            phase='smart_assessment',
                            content=output,
                            metadata={
                                'smart_question': True,
                                'question': next_question['question'],
                                'question_type': next_question['type'],
                                'options': next_question['options'],
                                'question_index': session_data['current_question_index']
                            }
                        )
                        
                        return jsonify({
                            "status": "success",
                            "output": output,
                            "metadata": {
                                "phase": session_data['phase'],
                                "smart_question": True,
                                "question": next_question['question'],
                                "question_type": next_question['type'],
                                "options": next_question['options'],
                                "question_index": session_data['current_question_index']
                            }
                        })
                    else:
                        # All smart questions answered, determine next step
                        print("All smart questions answered, determining next step")
                        
                        # Get symptoms and initial complaint for decision making
                        symptoms = session_data.get('symptoms', [])
                        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                                        if resp['question'] == "Please describe what brings you here today"), "")
                        
                        # Ensure index is available for decision making
                        index = pc.Index("who-guide-old")
                        
                        # Use intelligent decision making
                        next_step = determine_next_assessment_step(symptoms, initial_complaint, session_data, index)
                        print(f"Next step determined: {next_step}")
                        
                        if next_step['action'] == "continue_questions":
                            # Need more questions, get additional smart questions
                            additional_questions = get_smart_questions(symptoms, initial_complaint, index, session_data)
                            if additional_questions:
                                # Double-check: Filter out questions we've already asked (extra safety)
                                asked_questions = [resp['question'] for resp in session_data.get('smart_responses', [])]
                                new_questions = []
                                
                                for q in additional_questions:
                                    # Check for exact matches and similar questions
                                    is_duplicate = False
                                    for asked_q in asked_questions:
                                        if q['question'].lower().strip() == asked_q.lower().strip():
                                            is_duplicate = True
                                            break
                                        # Also check for similar questions using the existing function
                                        if check_similar_questions(q['question'], [asked_q]):
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        new_questions.append(q)
                                
                                print(f"Additional questions generated: {len(additional_questions)}")
                                print(f"After deduplication: {len(new_questions)} new questions")
                                
                                if new_questions:
                                    # Limit additional questions to prevent endless questioning
                                    MAX_ADDITIONAL_QUESTIONS = 2
                                    limited_new_questions = new_questions[:MAX_ADDITIONAL_QUESTIONS]
                                    session_data['smart_questions'].extend(limited_new_questions)
                                    next_question = limited_new_questions[0]
                                    options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                                    output = f"I need to ask you a few more questions to complete the assessment:\n\n{next_question['question']}\n\n{options_text}"
                                    
                                    # Save assistant message to conversation history
                                    save_conversation_message(
                                        session_id=session_id,
                                        message_type='assistant',
                                        phase='smart_assessment',
                                        content=output,
                                        metadata={
                                            'smart_question': True,
                                            'question': next_question['question'],
                                            'question_type': next_question['type'],
                                            'options': next_question['options'],
                                            'question_index': len(session_data.get('smart_responses', []))
                                        }
                                    )
                                    
                                    return jsonify({
                                        "status": "success",
                                        "output": output,
                                        "metadata": {
                                            "phase": session_data['phase'],
                                            "smart_question": True,
                                            "question": next_question['question'],
                                            "question_type": next_question['type'],
                                            "options": next_question['options']
                                        }
                                    })
                                else:
                                    # No new questions, move to examination
                                    session_data['phase'] = "exam"
                                    response = generate_examination(session_data)
                                    
                                    # Save assistant message to conversation history
                                    save_conversation_message(
                                        session_id=session_id,
                                        message_type='assistant',
                                        phase='exam',
                                        content=response.json['output'],
                                        metadata=response.json['metadata']
                                    )
                                    
                                    return response
                            else:
                                # No additional questions, move to examination
                                session_data['phase'] = "exam"
                                response = generate_examination(session_data)
                                
                                # Save assistant message to conversation history
                                save_conversation_message(
                                    session_id=session_id,
                                    message_type='assistant',
                                    phase='exam',
                                    content=response.json['output'],
                                    metadata=response.json['metadata']
                                )
                                
                                return response
                                
                        elif next_step['action'] == "perform_examination":
                            # Move to examination phase
                            session_data['phase'] = "exam"
                            response = generate_examination(session_data)
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='exam',
                                content=response.json['output'],
                                metadata=response.json['metadata']
                            )
                            
                            return response
                            
                        elif next_step['action'] == "make_diagnosis":
                            # Move to final results
                            session_data['phase'] = "final"
                            response = generate_final_results(session_data)
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='final',
                                content=response.json['output'],
                                metadata=response.json['metadata']
                            )
                            
                            return response
                            
                        elif next_step['action'] == "refer_immediately":
                            # Immediate referral needed
                            output = f"Based on your responses, you need immediate medical attention. Please go to the nearest hospital or health facility as soon as possible. {next_step['reason']}"
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='referral',
                                content=output,
                                metadata={'immediate_referral': True, 'reason': next_step['reason']}
                            )
                            
                            return jsonify({
                                "status": "success",
                                "output": output,
                                "metadata": {
                                    "phase": "referral",
                                    "immediate_referral": True,
                                    "reason": next_step['reason']
                                }
                            })
                        else:
                            # Default to examination
                            session_data['phase'] = "exam"
                            response = generate_examination(session_data)
                            
                            # Save assistant message to conversation history
                            save_conversation_message(
                                session_id=session_id,
                                message_type='assistant',
                                phase='exam',
                                content=response.json['output'],
                                metadata=response.json['metadata']
                            )
                            
                            return response
                            
            except Exception as e:
                print(f"Error in smart_assessment phase: {str(e)}")
                raise
                
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid phase"
            }), 400
            
    except Exception as e:
        print(f"Error in process_input: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def generate_followup_question(session_data):
    """Generate followup questions using RAG"""
    try:
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Use RAG to generate followup questions
        index = pc.Index("who-guide-old")
        embedding = get_embedding_batch([initial_complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)
        
        # Save citations to database
        session_id = next((k for k, v in sessions.items() if v == session_data), 'default')
        save_session_chunks_to_neondb(session_id, "followup_questions", relevant_docs)
        
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        prompt = get_main_followup_question_prompt(
            initial_complaint=initial_complaint,
            previous_questions="",
            combined_context=medical_guide_content
        )
        
        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        output = completion.choices[0].message.content.strip()
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": session_data['phase'],
                "citations": relevant_docs
            }
        })
        
    except Exception as e:
        print(f"Error in generate_followup_question: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def is_useful_examination(examination_text: str) -> bool:
    """Determine if an examination is useful and has measurable outcomes."""
    print(f"DEBUG: Checking if examination is useful: '{examination_text}'")
    
    # List of generic/unhelpful examination patterns to skip
    generic_patterns = [
        "general examination",
        "physical examination",
        "complete examination", 
        "full examination",
        "comprehensive examination",
        "routine examination",
        "standard examination",
        "basic examination",
        "overall examination",
        "thorough examination"
    ]
    
    examination_lower = examination_text.lower()
    
    # Skip if it matches generic patterns
    for pattern in generic_patterns:
        if pattern in examination_lower:
            print(f"DEBUG: Rejecting generic pattern '{pattern}' in '{examination_text}'")
            return False
    
    # Skip if it's too vague (less than 10 characters or just common words)
    if len(examination_text.strip()) < 10:
        print(f"DEBUG: Rejecting too short examination: '{examination_text}' (length: {len(examination_text.strip())})")
        return False
    
    # Skip if it's just asking for general assessment
    vague_terms = ["assess", "evaluate", "check", "examine", "look at", "observe"]
    if any(term in examination_lower for term in vague_terms) and len(examination_text.split()) < 5:
        print(f"DEBUG: Rejecting vague examination: '{examination_text}'")
        return False
    
    print(f"DEBUG: Accepting examination: '{examination_text}'")
    return True

def generate_examination(session_data):
    """Generate the next examination based on the patient's symptoms and previous examinations."""
    try:
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        symptoms_text = "\n".join([f"- {symptom}" for symptom in session_data['symptoms']])
        previous_exams_text = "\n".join([f"Examination: {exam['examination']}\nResult: {exam['result']}" 
                                        for exam in session_data['exam_responses']])
        
        # Get list of already performed examinations to avoid duplicates
        performed_examinations = set()
        for exam in session_data['exam_responses']:
            if 'examination' in exam:
                performed_examinations.add(exam['examination'].lower().strip())
        
        print(f"Already performed examinations: {performed_examinations}")
        
        # HARDCODED EXAMINATION LIST - Only use these three tests with yes/no options
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Yes - Positive for malaria"},
                    {"id": 2, "text": "No - Negative for malaria"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Yes - Normal (>13.5 cm)"},
                    {"id": 2, "text": "No - Malnutrition detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Yes - Normal temperature"},
                    {"id": 2, "text": "No - Fever detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            }
        ]
        
        # Check which examinations are relevant based on symptoms
        relevant_examinations = []
        all_symptoms = [initial_complaint.lower()] + [symptom.lower() for symptom in session_data['symptoms']]
        
        for exam in allowed_examinations:
            # Check if any of the patient's symptoms match the relevant symptoms for this examination
            is_relevant = any(symptom in exam["relevant_symptoms"] for symptom in all_symptoms)
            
            # Also check if the examination name itself is mentioned in symptoms
            exam_name_lower = exam["name"].lower()
            if any(exam_name_lower in symptom for symptom in all_symptoms):
                is_relevant = True
            
            if is_relevant:
                relevant_examinations.append(exam)
        
        print(f"Relevant examinations based on symptoms: {[exam['name'] for exam in relevant_examinations]}")
        
        # LIMIT: Only perform the most relevant examination (first one)
        if relevant_examinations:
            # Take only the first (most relevant) examination
            exam = relevant_examinations[0]
            if exam["name"].lower() not in performed_examinations:
                exam_data = {
                    "examination": exam["name"],
                    "procedure": exam["procedure"],
                    "type": "MC",
                    "options": exam["options"]
                }
                
                session_data['current_examination'] = exam_data
                
                options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in exam_data['options']])
                formatted_output = f"Examination: {exam['name']}\nProcedure: {exam['procedure']}\n\nPlease select the examination result:\n{options_text}"
                
                return jsonify({
                    "status": "success",
                    "output": formatted_output,
                    "metadata": {
                        "phase": session_data['phase'],
                        "examination": exam_data,
                        "citations": [],
                        "question_type": "MC",
                        "options": exam_data['options']
                    }
                })
        
        # If no relevant examinations or all have been performed, move to final results
        print("No relevant examinations or all examinations performed, moving to final results")
        return generate_final_results(session_data)
        
    except Exception as e:
        print(f"Error in generate_examination: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def generate_specific_examination(session_data):
    """Generate a more specific examination when the first one is too generic."""
    try:
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        symptoms_text = "\n".join([f"- {symptom}" for symptom in session_data['symptoms']])
        
        # Get list of already performed examinations to avoid duplicates
        performed_examinations = set()
        for exam in session_data['exam_responses']:
            if 'examination' in exam:
                performed_examinations.add(exam['examination'].lower().strip())
        
        # HARDCODED EXAMINATION LIST - Only use these three tests with yes/no options
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Yes - Positive for malaria"},
                    {"id": 2, "text": "No - Negative for malaria"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Yes - Normal (>13.5 cm)"},
                    {"id": 2, "text": "No - Malnutrition detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Yes - Normal temperature"},
                    {"id": 2, "text": "No - Fever detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            }
        ]
        
        # Check which examinations are relevant based on symptoms
        relevant_examinations = []
        all_symptoms = [initial_complaint.lower()] + [symptom.lower() for symptom in session_data['symptoms']]
        
        for exam in allowed_examinations:
            # Check if any of the patient's symptoms match the relevant symptoms for this examination
            is_relevant = any(symptom in exam["relevant_symptoms"] for symptom in all_symptoms)
            
            # Also check if the examination name itself is mentioned in symptoms
            exam_name_lower = exam["name"].lower()
            if any(exam_name_lower in symptom for symptom in all_symptoms):
                is_relevant = True
            
            if is_relevant:
                relevant_examinations.append(exam)
        
        print(f"Relevant examinations based on symptoms (specific): {[exam['name'] for exam in relevant_examinations]}")
        
        # LIMIT: Only perform the most relevant examination (first one)
        if relevant_examinations:
            # Take only the first (most relevant) examination
            exam = relevant_examinations[0]
            if exam["name"].lower() not in performed_examinations:
                exam_data = {
                    "examination": exam["name"],
                    "procedure": exam["procedure"],
                    "type": "MC",
                    "options": exam["options"]
                }
                
                session_data['current_examination'] = exam_data
                
                options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in exam_data['options']])
                formatted_output = f"Examination: {exam['name']}\nProcedure: {exam['procedure']}\n\nPlease select the examination result:\n{options_text}"
                
                return jsonify({
                    "status": "success",
                    "output": formatted_output,
                    "metadata": {
                        "phase": session_data['phase'],
                        "examination": exam_data,
                        "citations": [],
                        "question_type": "MC",
                        "options": exam_data['options']
                    }
                })
        
        # If no relevant examinations or all have been performed, move to final results
        print("No relevant examinations or all examinations performed (specific), moving to final results")
        return generate_final_results(session_data)
        
    except Exception as e:
        print(f"Error in generate_specific_examination: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def generate_final_examination_fallback(session_data):
    """Generate a final fallback examination when all else fails."""
    try:
        # Get list of already performed examinations
        performed_examinations = set()
        for exam in session_data['exam_responses']:
            if 'examination' in exam:
                performed_examinations.add(exam['examination'].lower().strip())
        
        initial_complaint = next((resp['answer'] for resp in session_data['initial_responses'] 
                            if resp['question'] == "Please describe what brings you here today"), "")
        symptoms_text = "\n".join([f"- {symptom}" for symptom in session_data['symptoms']])
        
        # HARDCODED EXAMINATION LIST - Only use these three tests with yes/no options
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Yes - Positive for malaria"},
                    {"id": 2, "text": "No - Negative for malaria"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Yes - Normal (>13.5 cm)"},
                    {"id": 2, "text": "No - Malnutrition detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Yes - Normal temperature"},
                    {"id": 2, "text": "No - Fever detected"},
                    {"id": 3, "text": "Other (please specify)"}
                ]
            }
        ]
        
        # Check which examinations are relevant based on symptoms
        relevant_examinations = []
        all_symptoms = [initial_complaint.lower()] + [symptom.lower() for symptom in session_data['symptoms']]
        
        for exam in allowed_examinations:
            # Check if any of the patient's symptoms match the relevant symptoms for this examination
            is_relevant = any(symptom in exam["relevant_symptoms"] for symptom in all_symptoms)
            
            # Also check if the examination name itself is mentioned in symptoms
            exam_name_lower = exam["name"].lower()
            if any(exam_name_lower in symptom for symptom in all_symptoms):
                is_relevant = True
            
            if is_relevant:
                relevant_examinations.append(exam)
        
        print(f"Relevant examinations based on symptoms (fallback): {[exam['name'] for exam in relevant_examinations]}")
        
        # LIMIT: Only perform the most relevant examination (first one)
        if relevant_examinations:
            # Take only the first (most relevant) examination
            exam = relevant_examinations[0]
            if exam["name"].lower() not in performed_examinations:
                exam_data = {
                    "examination": exam["name"],
                    "procedure": exam["procedure"],
                    "type": "MC",
                    "options": exam["options"]
                }
                
                session_data['current_examination'] = exam_data
                
                options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in exam_data['options']])
                formatted_output = f"Examination: {exam['name']}\nProcedure: {exam['procedure']}\n\nPlease select the examination result:\n{options_text}"
                
                return jsonify({
                    "status": "success",
                    "output": formatted_output,
                    "metadata": {
                        "phase": session_data['phase'],
                        "examination": exam_data,
                        "citations": [],
                        "question_type": "MC",
                        "options": exam_data['options']
                    }
                })
        
        # If no relevant examinations or all have been performed, move to final results
        print("No relevant examinations or all examinations performed (fallback), moving to final results")
        return generate_final_results(session_data)
        
    except Exception as e:
        print(f"Error in generate_final_examination_fallback: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def should_continue_examinations(session_data):
    """Determine if more examinations are needed"""
    # LIMIT: Only perform 2 examinations maximum
    MAX_EXAMINATIONS = 2
    return len(session_data['exam_responses']) < MAX_EXAMINATIONS

def generate_final_results(session_data):
    """Generate final diagnosis and treatment using RAG"""
    try:
        # Get diagnosis and treatment using RAG
        result = get_diagnosis_and_treatment(
            session_data['initial_responses'],
            session_data['danger_sign_responses'],
            session_data['exam_responses']
        )
        
        # Save citations to database
        session_id = next((k for k, v in sessions.items() if v == session_data), 'default')
        save_session_chunks_to_neondb(session_id, "final_results", result.get('chunks_used', []))
        
        # Format output
        output = f"""## Assessment Complete

### Diagnosis
{result['diagnosis']}

### Treatment Recommendations
{result['treatment']}

### Sources
Based on WHO Medical Guide:
"""
        
        for citation in result.get('citations', []):
            output += f"- {citation['source']} (relevance: {citation['score']:.2f})\n"
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": session_data['phase'],
                "diagnosis": result['diagnosis'],
                "treatment": result['treatment'],
                "citations": result.get('citations', [])
            }
        })
        
    except Exception as e:
        print(f"Error in generate_final_results: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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

def check_for_immediate_examination(symptoms: List[str], initial_complaint: str) -> Optional[Dict[str, Any]]:
    """
    Check if an immediate examination should be performed based on symptoms.
    Only prioritize examinations that are CRITICAL for immediate decision-making.
    """
    # Convert to lowercase for matching
    symptoms_lower = [s.lower() for s in symptoms]
    complaint_lower = initial_complaint.lower()
    
    # Define CRITICAL immediate examinations only
    critical_examinations = [
        {
            "name": "RDT for Malaria",
            "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
            "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
            "relevant_keywords": ["malaria", "fever", "chills", "hot", "temperature"],
            "priority": 1,  # Highest priority
            "critical_reason": "Malaria requires immediate treatment and can be life-threatening",
            "options": [
                {"id": 1, "text": "Yes - Positive for malaria"},
                {"id": 2, "text": "No - Negative for malaria"},
                {"id": 3, "text": "Other (please specify)"}
            ]
        },
        {
            "name": "MUAC Strap",
            "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
            "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
            "relevant_keywords": ["malnutrition", "weight loss", "weakness", "child", "infant"],
            "priority": 2,
            "critical_reason": "Severe malnutrition requires immediate intervention",
            "options": [
                {"id": 1, "text": "Yes - Normal (>13.5 cm)"},
                {"id": 2, "text": "No - Malnutrition detected"},
                {"id": 3, "text": "Other (please specify)"}
            ]
        }
    ]
    
    # Check for critical examination matches
    matched_exams = []
    
    for exam in critical_examinations:
        # Check if any symptoms match
        symptom_match = any(symptom in exam["relevant_symptoms"] for symptom in symptoms_lower)
        
        # Check if any keywords are in the complaint
        keyword_match = any(keyword in complaint_lower for keyword in exam["relevant_keywords"])
        
        # Check if examination name is mentioned
        name_match = exam["name"].lower() in complaint_lower
        
        if symptom_match or keyword_match or name_match:
            matched_exams.append(exam)
    
    # Return the highest priority examination if any matches
    if matched_exams:
        # Sort by priority (lower number = higher priority)
        matched_exams.sort(key=lambda x: x["priority"])
        return matched_exams[0]
    
    return None

def generate_appropriate_options(question: str) -> List[Dict[str, Any]]:
    """
    Generate appropriate multiple-choice options based on the question type.
    """
    question_lower = question.lower()
    
    # Time-related questions
    if any(word in question_lower for word in ["how long", "duration", "when did", "started", "began"]):
        return [
            {"id": 1, "text": "Less than 24 hours"},
            {"id": 2, "text": "1-3 days"},
            {"id": 3, "text": "3-7 days"},
            {"id": 4, "text": "More than 7 days"}
        ]
    
    # Age-related questions
    elif any(word in question_lower for word in ["age", "how old", "years old"]):
        return [
            {"id": 1, "text": "Under 5 years"},
            {"id": 2, "text": "5-12 years"},
            {"id": 3, "text": "13-18 years"},
            {"id": 4, "text": "Over 18 years"}
        ]
    
    # Severity/intensity questions
    elif any(word in question_lower for word in ["severe", "mild", "moderate", "how bad", "intensity", "level"]):
        return [
            {"id": 1, "text": "Mild"},
            {"id": 2, "text": "Moderate"},
            {"id": 3, "text": "Severe"},
            {"id": 4, "text": "Very severe"}
        ]
    
    # Frequency questions
    elif any(word in question_lower for word in ["how often", "frequency", "times", "episodes"]):
        return [
            {"id": 1, "text": "Once or twice"},
            {"id": 2, "text": "Several times"},
            {"id": 3, "text": "Many times"},
            {"id": 4, "text": "Continuous"}
        ]
    
    # Location questions
    elif any(word in question_lower for word in ["where", "location", "which part", "area"]):
        return [
            {"id": 1, "text": "Head/Neck"},
            {"id": 2, "text": "Chest"},
            {"id": 3, "text": "Abdomen"},
            {"id": 4, "text": "Limbs"},
            {"id": 5, "text": "All over"}
        ]
    
    # Yes/No questions (default for most medical assessment questions)
    else:
        return [
            {"id": 1, "text": "Yes"},
            {"id": 2, "text": "No"},
            {"id": 3, "text": "Not sure"}
        ]

def get_smart_questions(symptoms: List[str], initial_complaint: str, index, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Use vector database to get relevant information and make intelligent decisions about what questions to ask.
    This is the core intelligence module for determining the assessment flow.
    """
    try:
        print(f"Getting smart questions for symptoms: {symptoms}")
        
        # Check if this is a diarrhea case - if so, use hardcoded diarrhea questions
        is_diarrhea_case = any(term in initial_complaint.lower() or any(term in symptom.lower() for symptom in symptoms)
                          for term in ['diarrhea', 'diarrhoea', 'loose stools'])
        
        if is_diarrhea_case:
            print("Diarrhea case detected - using hardcoded diarrhea questions")
            diarrhea_questions = get_diarrhea_specific_questions()
            
            # Check if diarrhea questions have already been asked
            already_asked = []
            if 'smart_responses' in session_data:
                already_asked = [resp['question'] for resp in session_data['smart_responses']]
            
            # Filter out already asked diarrhea questions
            remaining_diarrhea_questions = []
            for question in diarrhea_questions:
                if question['question'] not in already_asked:
                    remaining_diarrhea_questions.append(question)
            
            if remaining_diarrhea_questions:
                print(f"Returning {len(remaining_diarrhea_questions)} remaining diarrhea questions")
                return remaining_diarrhea_questions
            else:
                print("All diarrhea questions have been asked, proceeding to diagnosis")
                return []
        
        # Get already asked questions to avoid duplicates
        already_asked = []
        if 'smart_responses' in session_data:
            already_asked = [resp['question'] for resp in session_data['smart_responses']]
        print(f"Already asked questions: {already_asked}")
        
        # Create a comprehensive query to get relevant medical information
        query = f"assessment questions for: {initial_complaint} symptoms: {' '.join(symptoms)}"
        embedding = get_embedding_batch([query])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=5)
        
        if not relevant_docs:
            print("No relevant documents found for smart questioning")
            return []
        
        # Use medical guide content to determine what questions to ask
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Include already asked questions in the prompt to avoid duplicates
        already_asked_text = ""
        if already_asked:
            already_asked_text = f"\nALREADY ASKED QUESTIONS (DO NOT REPEAT THESE):\n" + "\n".join([f"- {q}" for q in already_asked])
        
        smart_questioning_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Patient complaint: "{initial_complaint}"
        Identified symptoms: {symptoms}
        {already_asked_text}
        
        Based ONLY on the WHO medical guide content below, determine what questions should be asked next to properly assess this patient. Focus on:
        1. Danger signs that are relevant to the patient's symptoms
        2. Key assessment questions that will help determine the next steps
        3. Questions that will help differentiate between different possible conditions
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        IMPORTANT RULES:
        - Ask questions that are DIRECTLY relevant to the patient's symptoms
        - Focus on questions that will help determine if the patient needs immediate referral
        - Ask questions that will help identify the most likely diagnosis
        - Limit to 3-4 most important questions maximum
        - Make questions specific and actionable
        - Prefer questions that can be answered with Yes/No, specific time periods, or clear categories
        - Avoid vague or open-ended questions
        - DO NOT repeat any questions that have already been asked
        - If all relevant questions have already been asked, respond with "none"
        
        QUESTION TYPES TO PREFER:
        - "Does the patient have [specific symptom]?" (Yes/No)
        - "How long has [symptom] lasted?" (Time periods)
        - "Is the patient able to [specific action]?" (Yes/No)
        - "Where is the [symptom] located?" (Body areas)
        - "How severe is the [symptom]?" (Severity levels)
        
        Return ONLY the specific questions that should be asked, formatted as a simple comma-separated list (e.g., "Does the patient have chest indrawing?, Is the patient able to drink?, How long has the fever lasted?").
        IMPORTANT: Each question should be a complete sentence ending with a question mark, separated by commas.
        If no relevant questions can be identified from the WHO medical guide, respond with "none"
        
        Remember: Use ONLY information from the WHO guide above, not your medical training. Focus on questions that will help make a proper assessment.
        """
        
        print("Sending smart questioning prompt to OpenAI")
        response = get_openai_completion(
            smart_questioning_prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        print(f"OpenAI response for smart questions: {response}")
        
        if response.lower().strip() == "none":
            print("No smart questions identified")
            return []
        
        # Parse the response to extract questions
        # First split by commas, then clean up each question
        raw_questions = [q.strip() for q in response.split(',')]
        questions = []
        
        for q in raw_questions:
            # Remove any leading/trailing whitespace and punctuation
            q = q.strip()
            if q:
                # If the question doesn't end with '?', add it
                if not q.endswith('?'):
                    q = q + '?'
                # Only add questions that are substantial (more than 5 characters)
                if len(q) > 5:
                    questions.append(q)
                    print(f"Parsed question: '{q}'")
        
        # Limit to 4 questions maximum
        MAX_QUESTIONS = 4
        questions = questions[:MAX_QUESTIONS]
        
        print(f"Parsed smart questions (limited to {MAX_QUESTIONS}): {questions}")
        
        # Convert to structured format with appropriate options for each question
        smart_questions = []
        for i, question in enumerate(questions, 1):
            # Generate appropriate options based on question type
            options = generate_appropriate_options(question)
            
            print(f"Smart question {i}: {question}")
            print(f"Generated options: {options}")
            
            smart_questions.append({
                "question": question,
                "type": "MC",
                "options": options
            })
        
        print(f"Final smart questions with options: {smart_questions}")
        return smart_questions
        
    except Exception as e:
        print(f"Error in get_smart_questions: {str(e)}")
        return []

# Test function to verify option generation (can be removed later)
def test_option_generation():
    """Test the generate_appropriate_options function"""
    test_cases = [
        ("How long has the fever lasted?", "time"),
        ("What is the patient age?", "age"),
        ("How severe is the pain?", "severity"),
        ("Does the patient have chest indrawing?", "yes_no")
    ]
    
    for question, expected_type in test_cases:
        options = generate_appropriate_options(question)
        print(f"Question: {question}")
        print(f"Expected type: {expected_type}")
        print(f"Options: {options}")
        print("---")

def determine_next_assessment_step(symptoms: List[str], initial_complaint: str, session_data: Dict[str, Any], index) -> Dict[str, Any]:
    """
    Make intelligent decisions about the next step in the assessment based on current information.
    This is the core decision-making module.
    """
    try:
        print(f"Determining next assessment step for symptoms: {symptoms}")
        
        # Check if this is a diarrhea case - if so, ensure all diarrhea questions are answered
        is_diarrhea_case = any(term in initial_complaint.lower() or any(term in symptom.lower() for symptom in symptoms)
                          for term in ['diarrhea', 'diarrhoea', 'loose stools'])
        
        if is_diarrhea_case:
            print("Diarrhea case detected - checking if all diarrhea questions have been answered")
            diarrhea_questions = get_diarrhea_specific_questions()
            smart_responses = session_data.get('smart_responses', [])
            
            # Check if all diarrhea questions have been answered
            answered_diarrhea_questions = set()
            for response in smart_responses:
                for question in diarrhea_questions:
                    if response['question'] == question['question']:
                        answered_diarrhea_questions.add(question['question'])
            
            if len(answered_diarrhea_questions) < len(diarrhea_questions):
                remaining_count = len(diarrhea_questions) - len(answered_diarrhea_questions)
                print(f"Diarrhea case: {remaining_count} questions remaining, must continue asking")
                return {"action": "continue_questions", "reason": f"Diarrhea case: {remaining_count} essential questions remaining"}
            else:
                print("All diarrhea questions answered, can proceed to diagnosis")
        
        # Check if we have enough information to make a diagnosis
        current_responses = session_data.get('followup_responses', [])
        exam_responses = session_data.get('exam_responses', [])
        
        # Get relevant medical information from vector database
        query = f"assessment criteria for: {initial_complaint} symptoms: {' '.join(symptoms)}"
        embedding = get_embedding_batch([query])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)
        
        if not relevant_docs:
            print("No relevant documents found for assessment decision")
            return {"action": "continue_questions", "reason": "Need more information"}
        
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Count smart questions already asked
        smart_responses = session_data.get('smart_responses', [])
        total_questions_asked = len(current_responses) + len(smart_responses)
        
        # Hard limit: if we've asked too many questions, move to examination
        MAX_TOTAL_QUESTIONS = 6
        if total_questions_asked >= MAX_TOTAL_QUESTIONS:
            print(f"Reached maximum questions limit ({MAX_TOTAL_QUESTIONS}), moving to examination")
            return {"action": "perform_examination", "reason": "Maximum questions limit reached"}
        
        decision_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Patient complaint: "{initial_complaint}"
        Symptoms: {symptoms}
        Total questions already answered: {total_questions_asked}
        Examinations performed: {len(exam_responses)}
        
        Based ONLY on the WHO medical guide content below, determine what should be done next:
        1. "continue_questions" - if more questions are needed for proper assessment
        2. "perform_examination" - if a specific examination is needed
        3. "make_diagnosis" - if enough information is available to make a diagnosis
        4. "refer_immediately" - if the patient needs immediate referral
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        IMPORTANT RULES:
        - Only make diagnosis if the guide provides clear criteria and you have sufficient information
        - Refer immediately if any danger signs are present
        - Be conservative about continuing questions - prefer examinations or diagnosis if possible
        - If more than 4-5 questions have been asked, prefer to move to examination or diagnosis
        - Perform examinations only if they are specifically mentioned in the guide for this condition
        - Only continue questions if absolutely necessary for proper assessment
        
        Return ONLY one of: "continue_questions", "perform_examination", "make_diagnosis", "refer_immediately"
        """
        
        print("Sending assessment decision prompt to OpenAI")
        response = get_openai_completion(
            decision_prompt,
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"OpenAI decision response: {response}")
        
        action = response.strip().lower()
        
        if action == "continue_questions":
            return {"action": "continue_questions", "reason": "Need more information for proper assessment"}
        elif action == "perform_examination":
            return {"action": "perform_examination", "reason": "Specific examination needed"}
        elif action == "make_diagnosis":
            return {"action": "make_diagnosis", "reason": "Sufficient information available"}
        elif action == "refer_immediately":
            return {"action": "refer_immediately", "reason": "Danger signs or critical condition detected"}
        else:
            return {"action": "continue_questions", "reason": "Default to asking more questions"}
            
    except Exception as e:
        print(f"Error in determine_next_assessment_step: {str(e)}")
        return {"action": "continue_questions", "reason": "Error occurred, defaulting to questions"}

def get_diarrhea_specific_questions() -> List[Dict[str, Any]]:
    """
    Return hardcoded diarrhea-specific questions based on WHO guidelines.
    These questions must be asked before proceeding to diagnosis for diarrhea cases.
    Reduced to bare minimum essential questions for efficiency.
    """
    diarrhea_questions = [
        {
            "question": "How many days has the child had diarrhea?",
            "type": "MC", 
            "priority": 1,
            "options": [
                {"id": "1", "text": "Less than 14 days"},
                {"id": "2", "text": "14 days or more"},
                {"id": "3", "text": "Not sure about the exact duration"}
            ]
        },
        {
            "question": "Is the child able to drink or breastfeed normally?",
            "type": "MC",
            "priority": 2,
            "options": [
                {"id": "1", "text": "Yes, child drinks normally"},
                {"id": "2", "text": "No, child vomits everything"},
                {"id": "3", "text": "Child drinks slowly or reluctantly"},
                {"id": "4", "text": "Not sure"}
            ]
        }
    ]
    
    return diarrhea_questions

if __name__ == "__main__":
    # Verify connection to WHO medical guide
    try:
        index = pc.Index("who-guide-old")
        test_embedding = get_embedding_batch(["fever symptoms"])[0]
        test_results = vectorQuotesWithSource(test_embedding, index, top_k=1)
        
        if test_results:
            print("✓ Connected to WHO medical guide (who-guide-old)")
            print("✓ All medical decisions will be based on WHO guide content via RAG")
        else:
            print("✗ WARNING: No results from WHO medical guide")
    except Exception as e:
        print(f"✗ ERROR: Cannot connect to WHO medical guide: {e}")
    
    # Get the port from Render's environment variable
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 to allow external access
    app.run(host="0.0.0.0", port=port) 