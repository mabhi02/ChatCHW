from typing import Dict, List, Any, Optional, Tuple
import sys
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import torch
import torch.nn as nn

# Import MATRIX components
from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

# Load environment variables
load_dotenv()

# PostgreSQL/NeonDB Connection Setup
DB_URL = "postgresql://neondb_owner:npg_RvG4KaDUcOp9@ep-broad-frost-a5ovvl94-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

# Other API clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize MATRIX system and components
matrix = MATRIX()
decoder_tuner = DecoderTuner(matrix.meta_learner.decoder)
visualizer = AttentionVisualizer()
pattern_analyzer = PatternAnalyzer()

# Initial screening questions
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

# Database Functions

def setup_database():
    """Set up the PostgreSQL database schema."""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    try:
        # Create sessions table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            chat_name TEXT UNIQUE NOT NULL,
            initial_responses JSONB DEFAULT '[]',
            followup_responses JSONB DEFAULT '[]',
            exam_responses JSONB DEFAULT '[]',
            current_question_index INTEGER DEFAULT 0,
            phase TEXT DEFAULT 'initial',
            diagnosis TEXT,
            treatment TEXT
        )
        """)
        
        # Create questions table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id SERIAL PRIMARY KEY,
            chat_name TEXT NOT NULL,
            question TEXT NOT NULL,
            options JSONB,
            selected_option INTEGER,
            pattern JSONB,
            confidence FLOAT,
            selected_agent TEXT,
            weights JSONB,
            sources JSONB
        )
        """)
        
        # Create examinations table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS examinations (
            id SERIAL PRIMARY KEY,
            chat_name TEXT NOT NULL,
            examination TEXT NOT NULL,
            options JSONB,
            selected_option INTEGER
        )
        """)
        
        conn.commit()
        print("Database setup completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Database setup error: {e}")
    finally:
        cur.close()
        conn.close()

def get_db_connection():
    """Create a database connection."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn

def create_or_get_session(chat_name: str) -> Dict[str, Any]:
    """Create a new session or get an existing session for the given chat name."""
    conn = get_db_connection()
    try:
        # Use RealDictCursor to return results as dictionaries
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if session exists
            cur.execute("SELECT * FROM sessions WHERE chat_name = %s", (chat_name,))
            session = cur.fetchone()
            
            if not session:
                # Create new session
                cur.execute(
                    """
                    INSERT INTO sessions (chat_name)
                    VALUES (%s)
                    RETURNING *
                    """,
                    (chat_name,)
                )
                session = cur.fetchone()
                conn.commit()
            
            return dict(session)
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
        return {}
    finally:
        conn.close()

def update_session_responses(chat_name: str, response_type: str, response_data: Dict[str, Any]) -> None:
    """Update session responses."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First, get the current responses
            cur.execute(
                f"SELECT {response_type} FROM sessions WHERE chat_name = %s",
                (chat_name,)
            )
            result = cur.fetchone()
            
            if result:
                current_responses = result[0] if result[0] else []
                
                # Append the new response
                current_responses.append(response_data)
                
                # Update the database
                cur.execute(
                    f"UPDATE sessions SET {response_type} = %s WHERE chat_name = %s",
                    (Json(current_responses), chat_name)
                )
                conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
    finally:
        conn.close()

def update_session_phase(chat_name: str, phase: str) -> None:
    """Update session phase."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET phase = %s WHERE chat_name = %s",
                (phase, chat_name)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
    finally:
        conn.close()

def update_session_field(chat_name: str, field: str, value: Any) -> None:
    """Update a specific field in the session."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE sessions SET {field} = %s WHERE chat_name = %s",
                (value, chat_name)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Database error: {e}")
    finally:
        conn.close()

def store_question(chat_name: str, question_data: Dict[str, Any]) -> int:
    """Store a question in the questions table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Insert the question data
            cur.execute(
                """
                INSERT INTO questions 
                (chat_name, question, options, selected_option, pattern, confidence, selected_agent, weights, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    chat_name, 
                    question_data.get("question"), 
                    Json(question_data.get("options", [])),
                    question_data.get("selected_option"),
                    Json(question_data.get("pattern", [])),
                    question_data.get("confidence", 0.5),
                    question_data.get("selected_agent"),
                    Json(question_data.get("weights", [])),
                    Json(question_data.get("sources", []))
                )
            )
            question_id = cur.fetchone()[0]
            conn.commit()
            return question_id
    except Exception as e:
        conn.rollback()
        print(f"Database error in store_question: {e}")
        return -1
    finally:
        conn.close()

def store_examination(chat_name: str, examination_data: Dict[str, Any]) -> int:
    """Store examination data in the examinations table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Insert the examination data
            cur.execute(
                """
                INSERT INTO examinations
                (chat_name, examination, options, selected_option)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    chat_name,
                    examination_data.get("examination"),
                    Json(examination_data.get("options", [])),
                    examination_data.get("selected_option")
                )
            )
            examination_id = cur.fetchone()[0]
            conn.commit()
            return examination_id
    except Exception as e:
        conn.rollback()
        print(f"Database error in store_examination: {e}")
        return -1
    finally:
        conn.close()

def get_structured_questions(chat_name: str) -> List[Dict[str, Any]]:
    """Get all structured questions for a chat session."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM questions WHERE chat_name = %s ORDER BY id",
                (chat_name,)
            )
            questions = cur.fetchall()
            return [dict(q) for q in questions]
    except Exception as e:
        print(f"Database error in get_structured_questions: {e}")
        return []
    finally:
        conn.close()

def get_examination_history(chat_name: str) -> List[Dict[str, Any]]:
    """Get examination history for a chat session."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM examinations WHERE chat_name = %s ORDER BY id",
                (chat_name,)
            )
            examinations = cur.fetchall()
            return [dict(e) for e in examinations]
    except Exception as e:
        print(f"Database error in get_examination_history: {e}")
        return []
    finally:
        conn.close()

def get_session_responses(chat_name: str, response_type: str) -> List[Dict[str, Any]]:
    """Get specific responses for a chat session."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {response_type} FROM sessions WHERE chat_name = %s",
                (chat_name,)
            )
            result = cur.fetchone()
            if result and result[0]:
                return result[0]
            return []
    except Exception as e:
        print(f"Database error in get_session_responses: {e}")
        return []
    finally:
        conn.close()

def get_session_field(chat_name: str, field: str) -> Any:
    """Get a specific field from a session."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {field} FROM sessions WHERE chat_name = %s",
                (chat_name,)
            )
            result = cur.fetchone()
            if result:
                return result[0]
            return None
    except Exception as e:
        print(f"Database error in get_session_field: {e}")
        return None
    finally:
        conn.close()

# API Functions

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's API."""
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [[] for _ in texts]

def vectorQuotesWithSource(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search vector DB and return relevant matches with source information."""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [{
            "text": match['metadata']['text'],
            "id": match['id'],
            "source": match['metadata'].get('source', 'Unknown'),
            "score": match['score']
        } for match in results['matches']]
    except Exception as e:
        print(f"Error searching vector DB: {e}")
        return []

def get_openai_completion(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Get completion from OpenAI's GPT-4 API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return ""

# Core Logic Functions

def generate_question_with_options(chat_name: str) -> Dict[str, Any]:
    """
    Generate a follow-up question with options based on patient input.
    """
    try:
        # Get initial responses
        initial_responses = get_session_responses(chat_name, "initial_responses")
        followup_responses = get_session_responses(chat_name, "followup_responses")
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        context = f"Initial complaint: {initial_complaint}\n"
        if followup_responses:
            context += "Previous responses:\n"
            for resp in followup_responses:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        # Get embeddings and relevant documents
        index = pc.Index("final-asha")
        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            raise Exception("Could not generate relevant question")
        
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
        prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
        
        Previous questions asked:
        {previous_questions if followup_responses else "No previous questions yet"}
        
        Relevant medical context:
        {combined_context}
        
        Generate ONE focused, relevant follow-up question that is different from the previous questions.
        Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
        Do not as about compound questions like "Do you have fever and cough?" or "Do you have pain in your chest or abdomen?". It should be one or the other like "Do you have fever" or "Do you have pain in your chest?".
        There should be no "or" or "and" in the question as ask about one specific metric not compounded one.
        
        Return only the question text.'''
        
        question = get_openai_completion(
            prompt=prompt,
            max_tokens=150,
            temperature=0.3
        )
        
        # Generate options
        options_prompt = f'''Generate 4 concise answers for: "{question}"
        Clear, mutually exclusive options.
        Return each option on a new line (1-4).'''
        
        options_text = get_openai_completion(
            prompt=options_prompt,
            max_tokens=100,
            temperature=0.2
        )
        
        options = []
        for i, opt in enumerate(options_text.strip().split('\n')):
            if opt.strip():
                text = opt.strip()
                if text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                options.append({"id": i+1, "text": text})
        
        options.append({"id": 5, "text": "Other (please specify)"})
        
        # Use MATRIX to get pattern analysis
        matrix_output = process_with_matrix(question, followup_responses)
        
        return {
            "question": question,
            "options": options,
            "type": "MC",
            "matrix_output": matrix_output,
            "relevant_docs": relevant_docs
        }
        
    except Exception as e:
        print(f"Error generating question: {e}")
        
        # Return a fallback question if there's an error
        return {
            "question": "Are you experiencing any other symptoms?",
            "options": [
                {"id": 1, "text": "No other symptoms"},
                {"id": 2, "text": "Yes, mild additional symptoms"},
                {"id": 3, "text": "Yes, moderate additional symptoms"},
                {"id": 4, "text": "Yes, severe additional symptoms"},
                {"id": 5, "text": "Other (please specify)"}
            ],
            "type": "MC",
            "matrix_output": {
                "confidence": 0.5,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.5, "pessimist": 0.5}
            },
            "relevant_docs": []
        }

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                      context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced error handling."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        print(f"Pattern Analysis - Optimist: {patterns['optimist_confidence']:.2f}, "
              f"Pessimist: {patterns['pessimist_confidence']:.2f}")
        
        # Create standardized previous responses for MATRIX processing
        standardized_responses = []
        for resp in previous_responses:
            try:
                standardized_resp = {
                    "question": resp.get("question", ""),
                    "answer": resp.get("answer", ""),
                }
                if isinstance(standardized_resp["answer"], list):
                    standardized_resp["answer"] = ", ".join(standardized_resp["answer"])
                standardized_responses.append(standardized_resp)
            except Exception as e:
                print(f"Warning: Error standardizing response: {e}")
        
        state = matrix.state_encoder.encode_state(
            [],
            standardized_responses,
            current_text
        )
        
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.squeeze(1)
            
        optimist_view = matrix.optimist.evaluate(
            state,
            context_text if context_text else current_text
        )
        pessimist_view = matrix.pessimist.evaluate(
            state,
            context_text if context_text else current_text
        )
        
        matrix_output = matrix.process_state(
            [], 
            standardized_responses,
            current_text
        )
        
        return matrix_output
        
    except Exception as e:
        print(f"Warning: MATRIX processing error: {e}")
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5}
        }

def judge(chat_name: str, current_question: str) -> bool:
    """Judge if the current question is too similar using MATRIX."""
    followup_responses = get_session_responses(chat_name, "followup_responses")
    
    if len(followup_responses) >= MATRIXConfig.MAX_QUESTIONS:
        print("\nReached maximum number of questions.")
        return True
        
    if not followup_responses:
        return False
        
    try:
        matrix_output = process_with_matrix(
            current_question, 
            followup_responses
        )
        
        should_stop = (
            matrix_output["confidence"] > MATRIXConfig.SIMILARITY_THRESHOLD or
            len(followup_responses) >= MATRIXConfig.MAX_FOLLOWUPS or
            matrix_output["weights"]["optimist"] > 0.7
        )
        
        if should_stop:
            print("\nMATRIX suggests sufficient information gathered.")
            
        return should_stop
        
    except Exception as e:
        print(f"Warning: Similarity check falling back to basic method: {e}")
        return len(followup_responses) >= 5

def generate_examination(chat_name: str) -> Dict[str, Any]:
    """Generate and format an examination question."""
    try:
        # Get responses
        initial_responses = get_session_responses(chat_name, "initial_responses")
        followup_responses = get_session_responses(chat_name, "followup_responses")
        exam_responses = get_session_responses(chat_name, "exam_responses")
        
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get symptoms from followup responses
        symptoms = set()
        for resp in followup_responses:
            answer = resp['answer'].lower() if isinstance(resp['answer'], str) else ""
            for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                        'headache', 'nausea', 'dizziness', 'rash']:
                if symptom in answer:
                    symptoms.add(symptom)
        
        # Compress context for examination recommendations
        context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}
Previous findings: {str([exam.get('examination', '') for exam in exam_responses]) if exam_responses else "None"}"""

        # Get embeddings and relevant documents
        index = pc.Index("final-asha")
        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            raise Exception("Could not generate relevant examination")
        
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        # Modified prompt to include options in brackets
        prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {str([exam.get('examination', '') for exam in exam_responses]) if exam_responses else "None"}

Recommend ONE essential examination in this format (should not be first world exams like MRI, CT, Colonoscopy, etc.):

Examination: [Examination name]
Procedure: [Detailed procedure steps to perform the examination]

Finding options (each option should be on its own line in square brackets):
[First possible finding]
[Second possible finding]
[Third possible finding]
[Fourth possible finding]

IMPORTANT: Each option must be in square brackets exactly as shown above.
'''

        examination_text = get_openai_completion(
            prompt=prompt,
            max_tokens=500,
            temperature=0.2  # Reduced temperature for more consistent formatting
        )
        
        # Parse the examination text
        examination, option_texts = parse_examination_text(examination_text)
        
        # Create options list
        options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
        options.append({"id": 5, "text": "Other (please specify)"})
        
        return {
            "examination": examination,
            "examination_text": examination_text,
            "options": options,
            "type": "EXAM",
            "relevant_docs": relevant_docs
        }
        
    except Exception as e:
        print(f"Error generating examination: {e}")
        
        # Return a fallback examination if there's an error
        return {
            "examination": "General Physical Examination\nPerform a basic physical assessment including vital signs, general appearance, and basic systems examination.",
            "examination_text": "General Physical Examination\nPerform a basic physical assessment including vital signs, general appearance, and basic systems examination.\n#:Normal findings with no concerning signs or symptoms\n#:Mild abnormalities requiring monitoring\n#:Moderate abnormalities requiring further evaluation\n#:Severe abnormalities requiring immediate intervention",
            "options": [
                {"id": 1, "text": "Normal findings with no concerning signs or symptoms"},
                {"id": 2, "text": "Mild abnormalities requiring monitoring"},
                {"id": 3, "text": "Moderate abnormalities requiring further evaluation"},
                {"id": 4, "text": "Severe abnormalities requiring immediate intervention"},
                {"id": 5, "text": "Other (please specify)"}
            ],
            "type": "EXAM",
            "relevant_docs": []
        }

def judge_exam(chat_name: str, current_exam: str) -> bool:
    """Judge examination similarity with improved duplicate detection."""
    exam_responses = get_session_responses(chat_name, "exam_responses")
    
    if not exam_responses:
        return False
        
    try:
        if len(exam_responses) >= MATRIXConfig.MAX_EXAMS:
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
        
        for exam in exam_responses:
            prev_exam_lines = exam['examination'].split('\n') if isinstance(exam.get('examination'), str) else []
            prev_name = ""
            prev_procedure = ""
            
            for line in prev_exam_lines:
                if line.startswith("Examination:"):
                    prev_name = line.split('Examination:')[1].strip().lower()
                elif line.startswith("Procedure:"):
                    prev_procedure = line.split('Procedure:')[1].strip().lower()
            
            if (prev_name and current_exam_name and 
                (prev_name in current_exam_name or current_exam_name in prev_name)):
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
            exam_responses
        )
        
        should_end = (
            matrix_output["confidence"] > MATRIXConfig.EXAM_SIMILARITY_THRESHOLD or
            len(exam_responses) >= MATRIXConfig.MAX_EXAMS or
            matrix_output["weights"]["optimist"] > 0.8
        )
        
        if should_end:
            print("\nSufficient examinations completed based on comprehensive analysis.")
            
        return should_end
                
    except Exception as e:
        print(f"Warning: Exam similarity check falling back to basic method: {e}")
        return len(exam_responses) >= MATRIXConfig.MAX_EXAMS

def parse_examination_text(examination_text: str) -> tuple[str, list[str]]:
    """
    Parse examination text using "#:" delimiter with robust fallback options.
    Returns tuple of (examination text, list of options).
    """
    # Try parsing with "#:" delimiter first
    parts = examination_text.split("#:")
    
    if len(parts) >= 2:
        # First part is the examination text
        examination = parts[0].strip()
        
        # Remaining parts are options
        options = [opt.strip() for opt in parts[1:] if opt.strip()]
        
        return examination, options
    
    # Fallback: Look for options in brackets
    options = []
    lines = examination_text.strip().split("\n")
    examination_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for text in brackets which likely indicates options/findings
        if line.startswith("[") and line.endswith("]"):
            options.append(line[1:-1])  # Remove the brackets
        else:
            examination_lines.append(line)
    
    if options:
        return "\n".join(examination_lines), options
    
    # If we couldn't extract options, use generic ones
    examination = examination_text
    generic_options = [
        "Normal findings",
        "Abnormal findings", 
        "Inconclusive results",
        "Further examination needed"
    ]
    
    return examination, generic_options

def store_examination_data(chat_name: str, examination_text: str, selected_option: int):
    """Parse examination text and store in PostgreSQL."""
    try:
        # Parse examination and options
        examination, options = parse_examination_text(examination_text)
        
        # Create examination entry
        examination_entry = {
            "chat_name": chat_name,
            "examination": examination,
            "options": options,
            "selected_option": selected_option
        }
        
        # Store in PostgreSQL
        store_examination(chat_name, examination_entry)
        
    except Exception as e:
        print(f"Error storing examination: {e}")

def parse_question_data(chat_name: str, question: str, options: list, answer: str, 
                       matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format and store in PostgreSQL."""
    
    # Convert option list to formatted strings
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    
    # Get selected option index (1-based indexing)
    selected_idx = next((i+1 for i, opt in enumerate(options) 
                        if opt['text'] == answer), None)
    
    # Format sources from citations
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" 
              for cite in citations] if citations else []
    
    question_data = {
        "chat_name": chat_name,
        "question": question,
        "options": formatted_options,
        "selected_option": selected_idx,
        "pattern": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "confidence": matrix_output.get('confidence', 0.5),
        "selected_agent": matrix_output.get('selected_agent', 'optimist'),
        "weights": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "sources": sources
    }
    
    # Store in PostgreSQL
    question_id = store_question(chat_name, question_data)
    question_data["id"] = question_id
    
    return question_data

def get_diagnosis_and_treatment(chat_name: str) -> Dict[str, Any]:
    """Generate diagnosis and treatment recommendations."""
    try:
        # Get all responses from the database
        initial_responses = get_session_responses(chat_name, "initial_responses") 
        followup_responses = get_session_responses(chat_name, "followup_responses")
        exam_responses = get_session_responses(chat_name, "exam_responses")
        
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get only the most relevant findings for context
        key_findings = []
        for resp in followup_responses + exam_responses:
            if isinstance(resp.get('answer'), str):
                key_findings.append(f"{resp['answer']}")
            elif isinstance(resp.get('result'), str):
                key_findings.append(f"{resp['result']}")
        key_findings = key_findings[-3:]  # Only keep last 3 findings
        
        # First, get diagnosis using minimal context
        index = pc.Index("who-guide")
        diagnosis_embedding = get_embedding_batch([initial_complaint + " diagnosis"])[0]
        diagnosis_docs = vectorQuotesWithSource(diagnosis_embedding, index, top_k=2)
        
        diagnosis_context = " ".join([doc["text"] for doc in diagnosis_docs])
        short_diagnosis_prompt = f'''Patient complaint: {initial_complaint}
Key findings: {"; ".join(key_findings)}
Reference: {diagnosis_context[:200]}

List top 3-4 possible diagnoses based on symptoms.'''

        diagnosis = get_openai_completion(
            prompt=short_diagnosis_prompt,
            max_tokens=500,
            temperature=0.2
        )
        
        # Then, get treatment recommendations in separate calls
        treatment_parts = []
        treatment_docs = []
        
        # 1. Immediate Care
        immediate_embedding = get_embedding_batch([initial_complaint + " immediate care steps"])[0]
        immediate_docs = vectorQuotesWithSource(immediate_embedding, index, top_k=1)
        treatment_docs.extend(immediate_docs)
        
        if immediate_docs:
            immediate_prompt = f'''Based on: {immediate_docs[0]["text"][:200]}
Provide 2-3 immediate care steps for {initial_complaint}.'''
            
            immediate_care = get_openai_completion(
                prompt=immediate_prompt,
                max_tokens=500,
                temperature=0.2
            )
            treatment_parts.append("Immediate Care:\n" + immediate_care)
        
        # 2. Medications
        med_embedding = get_embedding_batch([initial_complaint + " medications treatment"])[0]
        med_docs = vectorQuotesWithSource(med_embedding, index, top_k=1)
        treatment_docs.extend(med_docs)
        
        if med_docs:
            med_prompt = f'''Based on: {med_docs[0]["text"][:200]}
List 2-3 key medications or supplements for {initial_complaint}.'''
            
            medications = get_openai_completion(
                prompt=med_prompt,
                max_tokens=500,
                temperature=0.2
            )
            treatment_parts.append("\nMedications/Supplements:\n" + medications)
        
        # 3. Home Care
        home_embedding = get_embedding_batch([initial_complaint + " home care follow up"])[0]
        home_docs = vectorQuotesWithSource(home_embedding, index, top_k=1)
        treatment_docs.extend(home_docs)
        
        if home_docs:
            home_prompt = f'''Based on: {home_docs[0]["text"][:200]}
List 2-3 home care instructions for {initial_complaint}.'''
            
            home_care = get_openai_completion(
                prompt=home_prompt,
                max_tokens=500,
                temperature=0.2
            )
            treatment_parts.append("\nHome Care:\n" + home_care)
        
        # Combine all parts
        treatment = "\n".join(treatment_parts)
        
        # Collect relevant citations
        citations = []
        citations.extend(diagnosis_docs)
        citations.extend(treatment_docs)
        
        # Update database with diagnosis and treatment
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions SET diagnosis = %s, treatment = %s, phase = %s
                    WHERE chat_name = %s
                    """,
                    (diagnosis, treatment, "complete", chat_name)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"Error updating diagnosis/treatment in DB: {e}")
        finally:
            conn.close()
        
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": citations
        }
            
    except Exception as e:
        print(f"Error in diagnosis/treatment: {e}")
        try:
            # Fallback to simpler request if the detailed one fails
            minimal_prompt = f"List possible diagnoses for: {initial_complaint}"
            fallback_diagnosis = get_openai_completion(
                prompt=minimal_prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            # Update database with fallback diagnosis
            conn = get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE sessions SET diagnosis = %s, treatment = %s, phase = %s
                        WHERE chat_name = %s
                        """,
                        (
                            fallback_diagnosis, 
                            "Please consult a healthcare provider for specific treatment recommendations.", 
                            "complete",
                            chat_name
                        )
                    )
                    conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Error updating fallback diagnosis in DB: {e}")
            finally:
                conn.close()
            
            return {
                "diagnosis": fallback_diagnosis,
                "treatment": "Please consult a healthcare provider for specific treatment recommendations.",
                "citations": []
            }
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return {
                "diagnosis": "Error generating diagnosis",
                "treatment": "Error generating treatment",
                "citations": []
            }

def handle_initial_question(chat_name: str, question_index: int, user_input: str) -> Dict[str, Any]:
    """Handle a response to an initial question."""
    try:
        # Get the current question
        current_question = questions_init[question_index]
        
        # Validate and process the response
        if current_question['type'] in ['MC', 'YN', 'MCM']:
            if current_question['type'] == 'MCM' and ',' in user_input:
                # Handle multiple selections
                answers = user_input.split(',')
                answers = [a.strip() for a in answers]
                
                # Check if all answers are valid
                all_valid = all(any(opt['id'] == int(a) for opt in current_question['options']) for a in answers if a.isdigit())
                
                if all_valid:
                    # Convert IDs to actual option text
                    selected_texts = []
                    for a in answers:
                        if a.isdigit():
                            opt_id = int(a)
                            for opt in current_question['options']:
                                if opt['id'] == opt_id:
                                    selected_texts.append(opt['text'])
                                    break
                    
                    # Store the response
                    response_data = {
                        "question": current_question['question'],
                        "answer": selected_texts,
                        "type": current_question['type']
                    }
                    update_session_responses(chat_name, "initial_responses", response_data)
                    
                    # Update question index
                    update_session_field(chat_name, "current_question_index", question_index + 1)
                    
                    # Return success
                    return {
                        "status": "success", 
                        "message": "Response stored successfully", 
                        "next_question_index": question_index + 1
                    }
                else:
                    return {"status": "error", "message": "Invalid multiple selection"}
            else:
                # Handle single selection
                try:
                    option_id = int(user_input.strip())
                    
                    # Check if option is valid
                    valid_option = any(opt['id'] == option_id for opt in current_question['options'])
                    
                    if valid_option:
                        # Get the selected option text
                        selected_text = next(opt['text'] for opt in current_question['options'] if opt['id'] == option_id)
                        
                        # Handle "Other" option
                        if option_id == 5 and selected_text == "Other (please specify)":
                            # For simplicity, we'll just store the custom text in place of the option
                            # In a real app, you might want the user to provide this text separately
                            selected_text = f"Other: {user_input}"
                        
                        # Store the response
                        response_data = {
                            "question": current_question['question'],
                            "answer": selected_text,
                            "type": current_question['type']
                        }
                        update_session_responses(chat_name, "initial_responses", response_data)
                        
                        # Update question index
                        update_session_field(chat_name, "current_question_index", question_index + 1)
                        
                        # Return success
                        return {
                            "status": "success", 
                            "message": "Response stored successfully", 
                            "next_question_index": question_index + 1
                        }
                    else:
                        return {"status": "error", "message": "Invalid option selected"}
                except ValueError:
                    return {"status": "error", "message": "Invalid input format"}
        
        elif current_question['type'] == 'NUM':
            try:
                # Parse and validate the numeric input
                num_value = int(user_input.strip())
                
                if current_question['range']['min'] <= num_value <= current_question['range']['max']:
                    # Store the response
                    response_data = {
                        "question": current_question['question'],
                        "answer": num_value,
                        "type": current_question['type']
                    }
                    update_session_responses(chat_name, "initial_responses", response_data)
                    
                    # Update question index
                    update_session_field(chat_name, "current_question_index", question_index + 1)
                    
                    # Return success
                    return {
                        "status": "success", 
                        "message": "Response stored successfully", 
                        "next_question_index": question_index + 1
                    }
                else:
                    return {
                        "status": "error", 
                        "message": f"Value must be between {current_question['range']['min']} and {current_question['range']['max']}"
                    }
            except ValueError:
                return {"status": "error", "message": "Invalid numeric input"}
        
        elif current_question['type'] == 'FREE':
            # Validate that the free text input is not empty
            if user_input.strip():
                # Store the response
                response_data = {
                    "question": current_question['question'],
                    "answer": user_input.strip(),
                    "type": current_question['type']
                }
                update_session_responses(chat_name, "initial_responses", response_data)
                
                # Update question index
                update_session_field(chat_name, "current_question_index", question_index + 1)
                
                # Return success
                return {
                    "status": "success", 
                    "message": "Response stored successfully", 
                    "next_question_index": question_index + 1
                }
            else:
                return {"status": "error", "message": "Response cannot be empty"}
        
        else:
            return {"status": "error", "message": f"Unsupported question type: {current_question['type']}"}
            
    except Exception as e:
        print(f"Error handling initial question: {e}")
        return {"status": "error", "message": str(e)}

def handle_followup_response(chat_name: str, user_input: str, current_question: Dict[str, Any]) -> Dict[str, Any]:
    """Process a response to a follow-up question."""
    try:
        # Validate the input based on the question options
        options = current_question.get('options', [])
        valid_ids = [str(opt['id']) for opt in options]
        
        # Check if input is a valid option ID
        if user_input in valid_ids:
            selected_id = int(user_input)
            
            # Get the corresponding text for the selected option
            selected_text = next(opt['text'] for opt in options if opt['id'] == selected_id)
            
            # Handle "Other" option (ID 5)
            if selected_id == 5:
                # In a real app, you would prompt for additional text
                # Here we'll just use a placeholder
                selected_text = "Other (user specified)"
            
            # Store the response in the database
            response_data = {
                "question": current_question['question'],
                "answer": selected_text,
                "type": "MC"
            }
            update_session_responses(chat_name, "followup_responses", response_data)
            
            # Store structured question data
            matrix_output = current_question.get('matrix_output', {})
            relevant_docs = current_question.get('relevant_docs', [])
            parse_question_data(
                chat_name,
                current_question['question'],
                options,
                selected_text,
                matrix_output,
                relevant_docs
            )
            
            # Check if we should move to the next phase
            if judge(chat_name, current_question['question']):
                # Update session phase to examination
                update_session_phase(chat_name, "examination")
                return {
                    "status": "success",
                    "message": "Response stored successfully",
                    "next_phase": "examination"
                }
            else:
                return {
                    "status": "success",
                    "message": "Response stored successfully",
                    "next_phase": "followup"
                }
        else:
            return {"status": "error", "message": "Invalid option selected"}
            
    except Exception as e:
        print(f"Error handling followup response: {e}")
        return {"status": "error", "message": str(e)}

def handle_examination_response(chat_name: str, user_input: str, current_examination: Dict[str, Any]) -> Dict[str, Any]:
    """Process a response to an examination question."""
    try:
        # Validate the input based on the examination options
        options = current_examination.get('options', [])
        valid_ids = [str(opt['id']) for opt in options]
        
        # Check if input is a valid option ID
        if user_input in valid_ids:
            selected_id = int(user_input)
            
            # Get the corresponding text for the selected option
            selected_text = next(opt['text'] for opt in options if opt['id'] == selected_id)
            
            # Handle "Other" option (ID 5)
            if selected_id == 5:
                # In a real app, you would prompt for additional text
                # Here we'll just use a placeholder
                selected_text = "Other (user specified)"
            
            # Store examination data in the database
            store_examination_data(chat_name, current_examination['examination_text'], selected_id)
            
            # Store the response in the session
            response_data = {
                "examination": current_examination['examination'],
                "result": selected_text,
                "type": "EXAM"
            }
            update_session_responses(chat_name, "exam_responses", response_data)
            
            # Check if we should move to the next phase
            if judge_exam(chat_name, current_examination['examination']):
                # Update session phase to diagnosis
                update_session_phase(chat_name, "diagnosis")
                return {
                    "status": "success",
                    "message": "Response stored successfully",
                    "next_phase": "diagnosis"
                }
            else:
                return {
                    "status": "success",
                    "message": "Response stored successfully",
                    "next_phase": "examination"
                }
        else:
            return {"status": "error", "message": "Invalid option selected"}
            
    except Exception as e:
        print(f"Error handling examination response: {e}")
        return {"status": "error", "message": str(e)}

# Initialization

def init():
    """Initialize the database and other components."""
    setup_database()
    print("Application initialized successfully.")

# Export all necessary functions
__all__ = [
    'setup_database',
    'create_or_get_session',
    'update_session_responses',
    'update_session_phase',
    'update_session_field',
    'store_question',
    'store_examination',
    'get_structured_questions',
    'get_examination_history',
    'get_session_responses',
    'get_session_field',
    'get_embedding_batch',
    'vectorQuotesWithSource',
    'get_openai_completion',
    'generate_question_with_options',
    'process_with_matrix',
    'judge',
    'generate_examination',
    'judge_exam',
    'parse_examination_text',
    'store_examination_data',
    'parse_question_data',
    'get_diagnosis_and_treatment',
    'handle_initial_question',
    'handle_followup_response',
    'handle_examination_response',
    'init',
    'questions_init'
]

# Initialize if this file is run directly
if __name__ == "__main__":
    init()