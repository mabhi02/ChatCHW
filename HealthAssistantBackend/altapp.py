from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import json
from typing import Dict, List, Any, Optional
import sys
import os
import openai
from pymongo import MongoClient
from bson.objectid import ObjectId
import psutil
import time
import threading
import logging
import os

# Import the updated functions from commandChad instead of chad
from newCommandChad import (
    questions_init,
    get_embedding_batch,
    vectorQuotesWithSource,
    process_with_matrix,
    judge,
    judge_exam,
    parse_examination_text,
    get_diagnosis_and_treatment,
    parse_question_data,
    store_examination_data,
    create_or_get_session,
    update_session_responses,
    update_session_phase,
    get_structured_questions,
    get_examination_history
)

app = Flask(__name__)
frontend_url = os.environ.get('FRONTEND_URL')
CORS(app, origins=["http://localhost:3000"])
openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# MongoDB Connection Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medical_assessment_CHW")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
sessions_collection = db["sessions"]
questions_collection = db["questions"]
examinations_collection = db["examinations"]

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    """Initialize a new session and return first question"""
    try:
        session_id = request.json.get('session_id', 'default')
        
        # Use the new MongoDB session management
        session = create_or_get_session(session_id)
        
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
        chat_name = data.get('session_id', 'default')
        user_input = data.get('input')
        
        print(f"Received input: {user_input} for session: {chat_name}")
        
        # Get session from MongoDB
        session = create_or_get_session(chat_name)
        print(f"Current phase: {session['phase']}")
        
        current_question_index = session.get('current_question_index', 0)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        exam_responses = session.get('exam_responses', [])
        
        if session['phase'] == "initial":
            try:
                # Store initial response
                current_question = questions_init[current_question_index]
                response_data = {
                    "question": current_question['question'],
                    "answer": user_input,
                    "type": current_question['type']
                }
                
                # Add to session in MongoDB
                update_session_responses(chat_name, "initial_responses", response_data)
                
                # Update current question index
                current_question_index += 1
                sessions_collection.update_one(
                    {"chat_name": chat_name},
                    {"$set": {"current_question_index": current_question_index}}
                )
                
                # Move to next question or phase
                if current_question_index < len(questions_init):
                    next_question = questions_init[current_question_index]
                    if next_question['type'] in ['MC', 'MCM']:
                        options_text = "\n".join([f"{opt['id']}. {opt['text']}" for opt in next_question['options']])
                        output = f"{next_question['question']}\n\n{options_text}"
                    else:
                        output = next_question['question']
                        
                    return jsonify({
                        "status": "success",
                        "output": output,
                        "metadata": {
                            "phase": session['phase'],
                            "question_type": next_question['type'],
                            "options": next_question.get('options', [])
                        }
                    })
                else:
                    print("Moving to followup phase")
                    update_session_phase(chat_name, "followup")
                    return generate_followup_question(chat_name)
                    
            except Exception as e:
                print(f"Error in initial phase: {str(e)}")
                raise
                
        elif session['phase'] == "followup":
            try:
                # Store followup response if there's a current question
                current_followup = session.get('current_followup_question', {})
                if current_followup:
                    response_data = {
                        "question": current_followup.get('question', ''),
                        "answer": user_input,
                        "type": "MC"
                    }
                    update_session_responses(chat_name, "followup_responses", response_data)
                
                # Get updated followup responses
                updated_session = create_or_get_session(chat_name)
                followup_responses = updated_session.get('followup_responses', [])
                
                # Generate next followup question or move to exam phase
                if judge(followup_responses, current_followup.get('question', '')):
                    update_session_phase(chat_name, "exam")
                    return generate_examination(chat_name)
                else:
                    return generate_followup_question(chat_name)
                    
            except Exception as e:
                print(f"Error in followup phase: {str(e)}")
                raise
                
        elif session['phase'] == "exam":
            try:
                # Store examination response if there's a current examination
                current_exam = session.get('current_examination', {})
                if current_exam:
                    # Find which option was selected by matching the text
                    selected_option = 1  # Default to first option
                    for i, option in enumerate(current_exam.get('options', []), 1):
                        if option.get('text') == user_input:
                            selected_option = i
                            break
                    
                    # Store in MongoDB using the new function
                    store_examination_data(chat_name, current_exam.get('text', ''), selected_option)
                    
                    # Add to exam responses
                    response_data = {
                        "examination": current_exam.get('text', ''),
                        "result": user_input,
                        "type": "EXAM"
                    }
                    update_session_responses(chat_name, "exam_responses", response_data)
                
                # Get updated exam responses
                updated_session = create_or_get_session(chat_name)
                exam_responses = updated_session.get('exam_responses', [])
                
                # Generate next examination or complete assessment
                if judge_exam(exam_responses, current_exam.get('text', '')):
                    return generate_final_results(chat_name)
                else:
                    return generate_examination(chat_name)
                    
            except Exception as e:
                print(f"Error in exam phase: {str(e)}")
                raise
                
    except Exception as e:
        print(f"Error in process_input: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

def generate_followup_question(chat_name):
    """Generate and format followup question using MongoDB session"""
    try:
        print("Starting generate_followup_question")
        
        # Get session data from MongoDB
        session = create_or_get_session(chat_name)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        print(f"Initial complaint: {initial_complaint}")
        
        context = f"Initial complaint: {initial_complaint}\n"
        if followup_responses:
            context += "Previous responses:\n"
            for resp in followup_responses:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        # Get embeddings and relevant documents
        index = pc.Index("final-asha")
        embedding = get_embedding_batch([context])[0]
        
        print("Got embedding")
        
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        print(f"Got {len(relevant_docs)} relevant docs")
        
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
        Follow standard medical assessment order:
        1. Duration and onset
        2. Characteristics and severity
        3. Associated symptoms
        4. Impact on daily life
        
        Return only the question text.'''
        
        print("Sending prompt to OpenAI")

        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        question = response.choices[0].message.content.strip()

        print(f"Got question: {question}")
        
        # Generate options
        options_prompt = f'''Generate 4 concise answers for: "{question}"
        Clear, mutually exclusive options.
        Return each option on a new line (1-4).'''
        
        options_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": options_prompt}
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
        
        print(f"Generated {len(options)} options")
        
        # Store the generated question in MongoDB session
        sessions_collection.update_one(
            {"chat_name": chat_name},
            {"$set": {"current_followup_question": {
                "question": question,
                "options": options
            }}}
        )
        
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
        print(f"Error in generate_followup_question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        })
     
def generate_examination(chat_name):
    """Generate examination using MongoDB session and robust parsing"""
    try:
        # Get session data from MongoDB
        session = create_or_get_session(chat_name)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        exam_responses = session.get('exam_responses', [])
        
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get embeddings for complaint and key symptoms
        context_items = [initial_complaint]
        symptoms = []
        for resp in followup_responses:
            answer = resp['answer'].lower() if isinstance(resp['answer'], str) else ""
            for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                          'headache', 'nausea', 'dizziness', 'rash']:
                if symptom in answer:
                    symptoms.append(answer)
                    context_items.append(answer)
        
        # Get embeddings for all context items
        embeddings = get_embedding_batch(context_items)
        
        # Get relevant documents using embeddings
        index = pc.Index("final-asha")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=1)
            if matches:
                relevant_matches.extend(matches)
        
        # Sort matches by relevance score
        relevant_matches.sort(key=lambda x: x['score'], reverse=True)
        top_match = relevant_matches[0] if relevant_matches else None
        
        # Build a prompt for examination that works better with the new parser
        symptoms_summary = ", ".join(symptoms[:3])
        
        prompt = f'''For a patient with: "{initial_complaint}"
Key symptoms: {symptoms_summary}
Most relevant condition (score {top_match["score"]:.2f}): {top_match["text"][:100] if top_match else "None"}

Generate ONE physical examination using EXACTLY this format:

Examination: [Examination name]
Procedure: [Detailed description of how to perform the examination]

Finding options (EACH OPTION MUST BE IN SQUARE BRACKETS):
[First possible finding - normal result]
[Second possible finding - mild abnormal result]
[Third possible finding - moderate abnormal result]
[Fourth possible finding - severe abnormal result]

IMPORTANT: Each finding option MUST be enclosed in square brackets.'''

        completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a medical AI. Always follow the exact format provided with findings in square brackets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.2
        )
        
        examination_text = completion.choices[0].message.content.strip()
        print(f"Generated examination text: {examination_text}")
        
        # Use the improved parser that handles different formats
        examination, option_texts = parse_examination_text(examination_text)
        
        options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts[:4])]
        options.append({"id": 5, "text": "Other (please specify)"})
        
        # Store in MongoDB session
        sessions_collection.update_one(
            {"chat_name": chat_name},
            {"$set": {"current_examination": {
                "text": examination_text,
                "options": options
            }}}
        )
        
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

def generate_final_results(chat_name):
    """Generate and format final results using MongoDB session"""
    try:
        # Get session data from MongoDB
        session = create_or_get_session(chat_name)
        initial_responses = session.get('initial_responses', [])
        followup_responses = session.get('followup_responses', [])
        exam_responses = session.get('exam_responses', [])
        
        results = get_diagnosis_and_treatment(
            initial_responses,
            followup_responses,
            exam_responses
        )
        
        # Update session with diagnosis and treatment
        sessions_collection.update_one(
            {"chat_name": chat_name},
            {"$set": {
                "diagnosis": results["diagnosis"],
                "treatment": results["treatment"],
                "phase": "complete"
            }}
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

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def log_memory_usage(interval=60):
    """Log memory usage at regular intervals"""
    while True:
        mem_usage = get_memory_usage()
        logging.info(f"Current memory usage: {mem_usage:.2f} MB")
        time.sleep(interval)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("memory_usage.log")
        ]
    )
    
    # Log initial memory usage
    initial_memory = get_memory_usage()
    logging.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Start memory tracking in a background thread
    memory_thread = threading.Thread(target=log_memory_usage, daemon=True)
    memory_thread.start()

    logging.info(f"Memory usage before server start: {get_memory_usage():.2f} MB")
    # Get the port from Render's environment variable
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 to allow external access
    app.run(host="0.0.0.0", port=port)