"""
This file fuses newCommandChad.py into app.py.
app.py: Web API interface (bridges backend logic with frontend using flask)
newCommandChad.py: adapts chatbot to work as a command-line application with persistend storage using NeonDB (PostgreSQL)
appNeonDB.py: Uses Web API interface but stores everything in the database


"""
# imports from newCommandChad
import psycopg2
from psycopg2.extras import RealDictCursor, Json

from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
import sys
import os
import openai
from chad import (
    questions_init,
    structured_questions_array,
    examination_history,
    get_embedding_batch,
    vectorQuotesWithSource,
    process_with_matrix,
    judge,
    judge_exam,
    parse_examination_text,
    get_diagnosis_and_treatment,
    parse_question_data,
    store_examination,
    initialize_session,
    get_session_data,
)


# PostgreSQL/NeonDB Connection Setup
DB_URL = "postgresql://neondb_owner:npg_RvG4KaDUcOp9@ep-broad-frost-a5ovvl94-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"

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


app = Flask(__name__)
# Set frontend URL with fallback to your deployed frontend
frontend_url = os.environ.get('FRONTEND_URL', 'https://chatchw.onrender.com')
# Configure CORS to allow requests from both localhost and your deployed frontend
CORS(app, origins=["http://localhost:3000", frontend_url], supports_credentials=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# Global state management
sessions = {}

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new session and return first question"""
    try:
        session_id = request.json.get('session_id', 'default')
        session_data = get_session_data(session_id, sessions)
        
        # Format first question for chat interface
        first_question = questions_init[0]
        question_text = first_question['question']
        
        if first_question['type'] in ['MC', 'MCM']:
            options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in first_question['options']])
            output = f"{question_text}\n\n{options_text}"
        else:
            output = question_text
            
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
                    return generate_followup_question(session_data)
                    
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
                    return generate_examination(session_data)
                else:
                    return generate_followup_question(session_data)
                    
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
                    
                    store_examination(session_data['current_examination']['text'], selected_option)
                    session_data['exam_responses'].append({
                        "examination": session_data['current_examination']['text'],
                        "result": user_input,
                        "type": "EXAM"
                    })
                
                # Generate next examination or complete assessment
                if judge_exam(session_data['exam_responses'], session_data['current_examination']['text']):
                    return generate_final_results(session_data)
                else:
                    return generate_examination(session_data)
                    
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
        
        print(f"Initial complaint: {initial_complaint}")  # Debug print
        
        context = f"Initial complaint: {initial_complaint}\n"
        if session_data['followup_responses']:
            context += "Previous responses:\n"
            for resp in session_data['followup_responses']:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        # Get embeddings and relevant documents
        index = pc.Index("who-guide")
        embedding = get_embedding_batch([context])[0]
        
        print("Got embedding")  # Debug print
        
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        print(f"Got {len(relevant_docs)} relevant docs")  # Debug print
        
        if not relevant_docs:
            raise Exception("Could not generate relevant question")
        
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        previous_questions = "\n".join([f"- {resp['question']}" for resp in session_data['followup_responses']])
        prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
        
        Previous questions asked:
        {previous_questions if session_data['followup_responses'] else "No previous questions yet"}
        
        Relevant medical context:
        {combined_context}

        Generate ONE focused, relevant follow-up question that is different from the previous questions.
        Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
        Do not as about compound questions like "Do you have fever and cough?" or "Do you have pain in your chest or abdomen?". It should be one or the other like "Do you have fever" or "Do you have pain in your chest?".
        There should be no "or" or "and" in the question as ask about one specific metric not compounded one.
                
        Return only the question text.
        '''
        
        print("Sending prompt to OpenAI")  # Debug print
        
        """
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        
        question = completion.choices[0].message.content.strip()
        """

        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",  # Using GPT-4-mini
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        question = response.choices[0].message.content.strip()


        print(f"Got question: {question}")  # Debug print
        
        # Generate options
        options_prompt = f'''Generate 4 concise answers for: "{question}"
        Clear, mutually exclusive options.
        Return each option on a new line (1-4).'''
        
        options_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",  # Using GPT-4-mini
            messages=[
                {"role": "system", "content": options_prompt}
            ],
            max_tokens=1500,
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
            "options": options
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
                "options": options
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
        
        # Get embeddings for complaint and key symptoms
        context_items = [initial_complaint]
        symptoms = []
        for resp in session_data['followup_responses']:
            answer = resp['answer'].lower()
            if any(symptom in answer for symptom in 
                  ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                   'headache', 'nausea', 'dizziness', 'rash']):
                symptoms.append(answer)
                context_items.append(answer)
        
        # Get embeddings for all context items
        embeddings = get_embedding_batch(context_items)
        
        # Get relevant documents using embeddings
        index = pc.Index("who-guide")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=1)
            if matches:
                relevant_matches.extend(matches)
        
        # Sort matches by relevance score
        relevant_matches.sort(key=lambda x: x['score'], reverse=True)
        top_match = relevant_matches[0] if relevant_matches else None
        
        # Build a compact prompt with very explicit formatting instructions
        symptoms_summary = ", ".join(symptoms[:3])
        
        prompt = f'''For a patient with: "{initial_complaint}"
Key symptoms: {symptoms_summary}
Most relevant condition (score {top_match["score"]:.2f}): {top_match["text"][:100] if top_match else "None"}

Generate ONE physical examination using EXACTLY this format (include the #: symbols before each finding):

YOUR EXAMINATION MUST:
1. Start with examination name
2. Then procedure description
3. Then EXACTLY 4 findings, each starting with #:
    a. # Finding 1
    b. # Finding 2
    c. # Finding 3
    d. # Finding 4
'''

        """
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a medical AI. Always follow the exact format provided with #: before each finding."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        """
        

        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",  # Using GPT-4-mini
            messages=[
                {"role": "system", "content": "You are a medical AI. Always follow the exact format provided with #: before each finding."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        examination_text = completion.choices[0].message.content.strip()
        print(f"Generated examination text: {examination_text}")  # Debug print
        
        examination, option_texts = parse_examination_text(examination_text)
        
        options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts[:4])]
        options.append({"id": 5, "text": "Other (please specify)"})
        
        session_data['current_examination'] = {
            "text": examination_text,
            "options": options
        }
        
        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in options])
        output = f"Recommended Examination:\n{examination}\n\nFindings:\n{options_text}"
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "exam",
                "question_type": "EXAM",
                "options": options
            }
        })
        
    except Exception as e:
        print(f"Error in generate_examination: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_final_results(session_data):
    """Generate and format final results"""
    try:
        results = get_diagnosis_and_treatment(
            session_data['initial_responses'],
            session_data['followup_responses'],
            session_data['exam_responses']
        )
        
        # Format results for chat interface
        output = f"""Assessment Complete

Diagnosis:
{results['diagnosis']}

Treatment Plan:
{results['treatment']}

Key References:
{chr(10).join([f"- {cite['source']} (relevance: {cite['score']:.2f})" for cite in results['citations']])}"""
        
        return jsonify({
            "status": "success",
            "output": output,
            "metadata": {
                "phase": "complete"
            }
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