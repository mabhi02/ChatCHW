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

# Initial screening questions (same as original)
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
        "type": "NUM",
        "range": {
            "min": 0,
            "max": 120,
            "step": 1,
            "unit": "years"
        }
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
        "type": "MCM",
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
    """Identify symptoms from the initial complaint using RAG from WHO guide."""
    try:
        print(f"Identifying symptoms from complaint: '{complaint}'")
        # Use RAG to identify symptoms mentioned in the complaint
        embedding = get_embedding_batch([complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=5)
        
        print(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            print("No relevant documents found")
            return [], []
        
        # Use medical guide content to identify symptoms
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        symptom_identification_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Based ONLY on the WHO medical guide content below, identify the main symptoms mentioned in this patient complaint: "{complaint}"
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        Return ONLY the specific symptoms that are:
        1. Mentioned in the patient's complaint
        2. Explicitly covered in the WHO medical guide content above
        
        IMPORTANT: 
        - If the patient mentions "fever", "hot", "temperature", or similar terms, and fever is mentioned in the WHO guide, include "fever"
        - If the patient mentions "chills", "cold", "shivering", or similar terms, and chills are mentioned in the WHO guide, include "chills"
        - If the patient mentions "cough", "coughing", or similar terms, and cough is mentioned in the WHO guide, include "cough"
        - If the patient mentions "diarrhea", "diarrhoea", "loose stools", or similar terms, and diarrhea is mentioned in the WHO guide, include "diarrhea"
        - If the patient mentions "vomit", "vomiting", "nausea", or similar terms, and vomiting is mentioned in the WHO guide, include "vomiting"
        
        Format your response as a simple comma-separated list of symptoms (e.g., "fever, cough, diarrhea")
        If no symptoms can be identified from the WHO medical guide, respond with "none"
        
        Remember: Use ONLY information from the WHO guide above, not your medical training.
        """
        
        print("Sending symptom identification prompt to OpenAI")
        response = get_openai_completion(
            prompt=symptom_identification_prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        print(f"OpenAI response: '{response}'")
        
        if response.lower().strip() == "none":
            print("No symptoms identified")
            return [], []
        
        # Parse the response to extract symptoms
        symptoms = [s.strip() for s in response.split(',')]
        symptoms = [s for s in symptoms if s and len(s) > 2]
        
        print(f"Parsed symptoms: {symptoms}")
        return symptoms, relevant_docs
        
    except Exception as e:
        print(f"Error identifying symptoms from complaint: {e}")
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
    """Check if a new question is similar to any previously asked questions using GPT."""
    if not previous_questions:
        return False
    
    try:
        # Create a list of previous questions for comparison
        previous_questions_text = "\n".join([f"- {q}" for q in previous_questions])
        
        prompt = f"""
        You are a medical assistant checking for duplicate or similar questions.
        
        Previous questions asked:
        {previous_questions_text}
        
        New question: {new_question}
        
        Determine if the new question is asking about the same medical issue as any of the previous questions.
        Consider synonyms and different ways of asking the same thing.
        
        Examples of similar questions:
        - "Is the patient unable to drink or eat anything?" vs "Does the patient have trouble drinking and feeding?"
        - "Does the patient have chest pain?" vs "Is the patient experiencing chest discomfort?"
        - "Does the patient have fever?" vs "Is the patient running a temperature?"
        
        Respond with ONLY "YES" if the new question is similar to any previous question, or "NO" if it's different.
        """
        
        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a medical assistant that identifies duplicate or similar medical questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        response = completion.choices[0].message.content.strip().upper()
        is_similar = response == "YES"
        
        print(f"Checking similarity: '{new_question}' vs {len(previous_questions)} previous questions")
        print(f"GPT response: {response} (similar: {is_similar})")
        
        return is_similar
        
    except Exception as e:
        print(f"Error checking question similarity: {e}")
        # Fallback: simple keyword matching
        new_question_lower = new_question.lower()
        for prev_question in previous_questions:
            prev_lower = prev_question.lower()
            # Check for common medical terms that might indicate similarity
            common_terms = ['drink', 'eat', 'feeding', 'fever', 'temperature', 'pain', 'chest', 'breathing']
            for term in common_terms:
                if term in new_question_lower and term in prev_lower:
                    return True
        return False

def ask_danger_sign_questions(symptoms: List[str], index, session_data=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Ask danger sign questions for each symptom using RAG from WHO guide."""
    danger_sign_responses = []
    all_citations = []
    
    print(f"Generating danger sign questions for symptoms: {symptoms}")
    
    # Get patient age from session data (default to 30 if not available)
    patient_age = 30
    if session_data and 'initial_responses' in session_data:
        for response in session_data['initial_responses']:
            if response['question'] == "What is the patient's age?":
                try:
                    patient_age = int(response['answer'])
                    print(f"Extracted patient age: {patient_age}")
                except (ValueError, TypeError):
                    patient_age = 30
                    print(f"Could not parse age '{response['answer']}', using default: {patient_age}")
                break
    
    # Get patient sex from session data (default to 'Unknown' if not available)
    patient_sex = 'Unknown'
    if session_data and 'initial_responses' in session_data:
        for response in session_data['initial_responses']:
            if response['question'] == "What is the patient's sex?":
                patient_sex = response['answer']
                print(f"Extracted patient sex: {patient_sex}")
                break
    
    # Combine all symptoms into a single complaint
    combined_complaint = " and ".join(symptoms)
    print(f"Combined complaint for danger sign search: {combined_complaint}")
    
    try:
        # Use the proven working function
        danger_signs_result = identify_danger_signs(combined_complaint, patient_age, patient_sex)
        print(f"Danger signs result: {danger_signs_result}")
        
        # Extract danger signs from the result
        if danger_signs_result and 'danger_signs' in danger_signs_result:
            danger_signs = danger_signs_result['danger_signs']
            print(f"Found {len(danger_signs)} danger signs: {danger_signs}")
            
            # Convert danger signs to questions with proper options (deduplicated)
            seen_questions = set()
            seen_danger_signs = set()  # Track the actual danger sign text to avoid duplicates
            
            for danger_sign in danger_signs:
                if isinstance(danger_sign, dict) and 'danger_sign' in danger_sign:
                    danger_sign_text = danger_sign['danger_sign']
                elif isinstance(danger_sign, str):
                    danger_sign_text = danger_sign
                else:
                    continue
                
                # Skip if we've already seen this exact danger sign
                if danger_sign_text.lower() in seen_danger_signs:
                    print(f"Skipping duplicate danger sign: {danger_sign_text}")
                    continue
                seen_danger_signs.add(danger_sign_text.lower())
                
                # Rephrase the danger sign into a coherent question
                question = rephrase_danger_sign_question(danger_sign_text)
                print(f"Rephrased question: {question}")
                
                # Skip if we've already seen this question
                if question.lower() in seen_questions:
                    print(f"Skipping duplicate question: {question}")
                    continue
                seen_questions.add(question.lower())
                
                # Check for similar questions using GPT
                if session_data and 'followup_responses' in session_data:
                    previous_questions = []
                    for response in session_data['followup_responses']:
                        if 'question' in response:
                            previous_questions.append(response['question'])
                    
                    if check_similar_questions(question, previous_questions):
                        print(f"Skipping similar question (GPT detected): {question}")
                        continue
                
                danger_sign_responses.append({
                    'symptom': symptoms[0] if symptoms else 'unknown',
                    'question': question,
                    'answer': None,
                    'type': 'MC',
                    'options': [
                        {"id": 1, "text": "Yes"},
                        {"id": 2, "text": "No"},
                        {"id": 3, "text": "Not sure"}
                    ]
                })
        
        # Get citations from the result if available
        if danger_signs_result and 'citations' in danger_signs_result:
            all_citations.extend(danger_signs_result['citations'])
            
    except Exception as e:
        print(f"Error using identify_danger_signs: {e}")
    
    print(f"Total danger sign responses generated: {len(danger_sign_responses)}")
    return danger_sign_responses, all_citations

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
        else:
            output = first_question['question']
        
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
                "phase": session_data['phase'],
                "question_type": first_question['type'],
                "options": first_question.get('options', [])
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
                    
                    # Generate danger sign questions
                    print(f"Identified symptoms: {symptoms}")
                    danger_sign_responses, danger_citations = ask_danger_sign_questions(symptoms, index, session_data)
                    print(f"Generated {len(danger_sign_responses)} danger sign questions")
                    session_data['danger_sign_responses'] = danger_sign_responses
                    session_data['citations'].extend(danger_citations)
                    
                    # Save danger sign citations to database
                    save_session_chunks_to_neondb(session_id, "danger_signs", danger_citations)
                    
                    if danger_sign_responses:
                        # Ask first danger sign question
                        first_danger_question = danger_sign_responses[0]
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_danger_question['options']])
                        output = f"Based on your symptoms, I need to check for any danger signs. Please answer the following question:\n\n{first_danger_question['question']}\n\n{options_text}"
                        
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
                        # No danger signs found, generate a simple followup question instead
                        print("No danger signs found, generating followup question")
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
                print(f"Error in initial phase: {str(e)}")
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
        
        # HARDCODED EXAMINATION LIST - Only use these three tests
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Positive for malaria"},
                    {"id": 2, "text": "Negative for malaria"},
                    {"id": 3, "text": "Invalid test result"},
                    {"id": 4, "text": "Unable to perform test"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Normal (>13.5 cm)"},
                    {"id": 2, "text": "Moderate malnutrition (11.5-13.5 cm)"},
                    {"id": 3, "text": "Severe malnutrition (<11.5 cm)"},
                    {"id": 4, "text": "Unable to measure"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Normal temperature (36.5-37.5°C)"},
                    {"id": 2, "text": "Low fever (37.6-38.5°C)"},
                    {"id": 3, "text": "High fever (>38.5°C)"},
                    {"id": 4, "text": "Hypothermia (<36.5°C)"},
                    {"id": 5, "text": "Unable to measure"}
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
        
        # Find the next examination that hasn't been performed
        for exam in relevant_examinations:
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
        
        # If all relevant examinations have been performed, move to final results
        print("All relevant examinations have been performed, moving to final results")
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
        
        # HARDCODED EXAMINATION LIST - Only use these three tests
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Positive for malaria"},
                    {"id": 2, "text": "Negative for malaria"},
                    {"id": 3, "text": "Invalid test result"},
                    {"id": 4, "text": "Unable to perform test"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Normal (>13.5 cm)"},
                    {"id": 2, "text": "Moderate malnutrition (11.5-13.5 cm)"},
                    {"id": 3, "text": "Severe malnutrition (<11.5 cm)"},
                    {"id": 4, "text": "Unable to measure"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Normal temperature (36.5-37.5°C)"},
                    {"id": 2, "text": "Low fever (37.6-38.5°C)"},
                    {"id": 3, "text": "High fever (>38.5°C)"},
                    {"id": 4, "text": "Hypothermia (<36.5°C)"},
                    {"id": 5, "text": "Unable to measure"}
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
        
        # Find the next examination that hasn't been performed
        for exam in relevant_examinations:
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
        
        # If all relevant examinations have been performed, move to final results
        print("All relevant examinations have been performed (specific), moving to final results")
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
        
        # HARDCODED EXAMINATION LIST - Only use these three tests
        allowed_examinations = [
            {
                "name": "RDT for Malaria",
                "procedure": "Use a rapid diagnostic test kit to detect malaria parasites in the patient's blood. Follow the kit instructions carefully.",
                "relevant_symptoms": ["fever", "chills", "malaria", "headache", "body aches", "sweating"],
                "options": [
                    {"id": 1, "text": "Positive for malaria"},
                    {"id": 2, "text": "Negative for malaria"},
                    {"id": 3, "text": "Invalid test result"},
                    {"id": 4, "text": "Unable to perform test"}
                ]
            },
            {
                "name": "MUAC Strap",
                "procedure": "Measure the Mid-Upper Arm Circumference using a MUAC strap. Wrap the strap around the patient's left arm, midway between the shoulder and elbow. Read the measurement at the arrow.",
                "relevant_symptoms": ["malnutrition", "weight loss", "poor appetite", "weakness", "child", "infant"],
                "options": [
                    {"id": 1, "text": "Normal (>13.5 cm)"},
                    {"id": 2, "text": "Moderate malnutrition (11.5-13.5 cm)"},
                    {"id": 3, "text": "Severe malnutrition (<11.5 cm)"},
                    {"id": 4, "text": "Unable to measure"}
                ]
            },
            {
                "name": "Thermometer",
                "procedure": "Use a digital thermometer to measure the patient's body temperature. Place the thermometer under the tongue or in the armpit for 2-3 minutes.",
                "relevant_symptoms": ["fever", "chills", "hot", "temperature", "sweating", "infection"],
                "options": [
                    {"id": 1, "text": "Normal temperature (36.5-37.5°C)"},
                    {"id": 2, "text": "Low fever (37.6-38.5°C)"},
                    {"id": 3, "text": "High fever (>38.5°C)"},
                    {"id": 4, "text": "Hypothermia (<36.5°C)"},
                    {"id": 5, "text": "Unable to measure"}
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
        
        # Find the next examination that hasn't been performed
        for exam in relevant_examinations:
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
        
        # If all relevant examinations have been performed, move to final results
        print("All relevant examinations have been performed (fallback), moving to final results")
        return generate_final_results(session_data)
        
    except Exception as e:
        print(f"Error in generate_final_examination_fallback: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def should_continue_examinations(session_data):
    """Determine if more examinations are needed"""
    # Simple logic: continue if less than 3 examinations
    return len(session_data['exam_responses']) < 3

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