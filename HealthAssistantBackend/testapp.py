import os
import uuid
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import time
from functools import wraps
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import all necessary components
from groq import Groq
from pinecone import Pinecone
import openai
import torch

from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('medical_api')

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler('medical_api.log', maxBytes=10000000, backupCount=5)

# Create formatters and add it to handlers
log_format = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(funcName)s:%(lineno)d'
)
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize environment and clients
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize MATRIX components
matrix = MATRIX()
decoder_tuner = DecoderTuner(matrix.meta_learner.decoder)
visualizer = AttentionVisualizer()
pattern_analyzer = PatternAnalyzer()

app = Flask(__name__)
CORS(app)

# Store conversation state
conversations: Dict[str, Dict[str, Any]] = {}

def log_execution_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{f.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

def log_api_call(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"API call started: {f.__name__}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        try:
            result = f(*args, **kwargs)
            end_time = time.time()
            logger.info(f"API call completed: {f.__name__}")
            logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"API call failed: {f.__name__}")
            logger.error(f"Error: {str(e)}")
            raise
    return wrapper

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's API."""
    logger.info(f"Getting embeddings for {len(texts)} texts")
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return [[] for _ in texts]

def vectorQuotesWithSource(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search vector DB and return relevant matches with source information."""
    logger.info(f"Searching vector DB for top {top_k} matches")
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
        logger.error(f"Error searching vector DB: {e}")
        return []

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/initial_questions", methods=["GET"])
def get_initial_questions():
    """Return initial screening questions."""
    questions = [
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
    return jsonify(questions)

@app.route("/chat", methods=["POST"])
@log_api_call
@log_execution_time
def chat():
    """Enhanced chat endpoint with comprehensive processing."""
    logger.info("Received chat request")
    try:
        data = request.json or {}
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        message = data.get("message", "")
        conversation_id = data.get("conversationId")
        screening_answers = data.get("screeningAnswers", [])
        conversation_history = data.get("conversationHistory", [])
        
        logger.info(f"Processing request with conversation_id={conversation_id}")

        # If we have screening answers, this is the initial submission
        if screening_answers:
            logger.info("Processing initial screening answers")
            try:
                # Get the initial complaint from screening answers
                initial_complaint = next((
                    answer["answer"] for answer in screening_answers 
                    if answer["question"] == "Please describe what brings you here today"
                ), "")
                
                logger.info(f"Initial complaint: {initial_complaint}")
                
                # Generate first follow-up question immediately
                logger.info("Generating first follow-up question")
                context = f"Initial complaint: {initial_complaint}"
                
                # Get relevant documents
                embedding = get_embedding_batch([context])[0]
                index = pc.Index("final-asha")
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                question_prompt = f'''Based on initial complaint: "{initial_complaint}"
Medical context: {combined_context}

Generate ONE focused follow-up question focusing on:
1. Duration and onset
2. Characteristics and severity
3. Associated symptoms
4. Impact on daily life

Return only the question text.'''

                completion = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": question_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=150
                )
                
                question = completion.choices[0].message.content.strip()
                logger.info(f"Generated question: {question}")
                
                # Generate options
                options_prompt = f'''For the medical question: "{question}"
Generate 4 clear, clinically relevant answer options.
Options should be mutually exclusive and cover likely scenarios.
Return each option on a new line (1-4).'''

                options_completion = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": options_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=100
                )
                
                options = []
                for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
                    if opt.strip():
                        text = opt.strip()
                        if text[0].isdigit() and text[1] in ['.', '-', ')']:
                            text = text[2:].strip()
                        options.append({"id": i + 1, "text": text})
                        
                options.append({"id": 5, "text": "Other (please specify)"})
                
                # Create new conversation with generated content
                new_conversation_id = str(uuid.uuid4())
                conversations[new_conversation_id] = {
                    "id": new_conversation_id,
                    "initial_responses": screening_answers,
                    "followup_responses": [],
                    "examination_responses": [],
                    "structured_questions": [],
                    "examination_history": [],
                    "citations": relevant_docs[-5:],
                    "stage": "followup",
                    "matrix_outputs": [],
                    "current_question": {
                        "question": question,
                        "options": options
                    }
                }
                
                response = {
                    "type": "QUESTION",
                    "question": question,
                    "options": options,
                    "citations": relevant_docs[-5:],
                    "conversationId": new_conversation_id
                }
                
                logger.info(f"Sending response: {json.dumps(response, indent=2)}")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error processing screening answers: {str(e)}")
                return jsonify({
                    "type": "ERROR",
                    "error": "Error generating follow-up question",
                    "message": str(e),
                    "options": [
                        {"id": "retry", "text": "Try again"},
                        {"id": "restart", "text": "Start over"}
                    ]
                })

        # Handle follow-up responses
        elif conversation_id and message:
            if conversation_id not in conversations:
                logger.error(f"Conversation {conversation_id} not found")
                return jsonify({
                    "type": "ERROR",
                    "error": "Conversation not found",
                    "options": [{"id": "restart", "text": "Start over"}]
                })
            
            conversation = conversations[conversation_id]
            logger.info(f"Processing follow-up response for conversation {conversation_id}")
            
            try:
                # Store the response
                conversation["followup_responses"].append({
                    "question": conversation["current_question"]["question"],
                    "answer": message
                })
                
                # Get context from conversation history
                context = f"Initial complaint: {conversation['initial_responses'][-1]['answer']}\n"
                context += "Previous responses:\n"
                for resp in conversation["followup_responses"]:
                    context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
                
                # Get relevant documents
                embedding = get_embedding_batch([context])[0]
                index = pc.Index("final-asha")
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                # Generate next question
                question_prompt = f'''Based on conversation history: {context}
Medical context: {combined_context}

Generate ONE focused follow-up question focusing on:
1. Duration and onset
2. Characteristics and severity
3. Associated symptoms
4. Impact on daily life

Make sure the question is different from previous ones.
Return only the question text.'''

                completion = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": question_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=150
                )
                
                question = completion.choices[0].message.content.strip()
                
                # Generate options
                options_prompt = f'''For the medical question: "{question}"
Generate 4 clear, clinically relevant answer options.
Options should be mutually exclusive and cover likely scenarios.
Return each option on a new line (1-4).'''

                options_completion = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": options_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=100
                )
                
                options = []
                for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
                    if opt.strip():
                        text = opt.strip()
                        if text[0].isdigit() and text[1] in ['.', '-', ')']:
                            text = text[2:].strip()
                        options.append({"id": i + 1, "text": text})
                        
                options.append({"id": 5, "text": "Other (please specify)"})
                
                # Update conversation state
                conversation["current_question"] = {
                    "question": question,
                    "options": options
                }
                
                response = {
                    "type": "QUESTION",
                    "question": question,
                    "options": options,
                    "citations": relevant_docs[-5:],
                    "conversationId": conversation_id
                }
                
                logger.info(f"Sending follow-up response: {json.dumps(response, indent=2)}")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error generating follow-up question: {str(e)}")
                return jsonify({
                    "type": "ERROR",
                    "error": "Error generating follow-up question",
                    "message": str(e),
                    "options": [
                        {"id": "retry", "text": "Try again"},
                        {"id": "restart", "text": "Start over"}
                    ],
                    "conversationId": conversation_id
                })
                
        else:
            logger.error("Invalid request format")
            return jsonify({
                "type": "ERROR",
                "error": "Invalid request format",
                "options": [{"id": "restart", "text": "Start over"}]
            })
            
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            "type": "ERROR",
            "error": "An unexpected error occurred",
            "message": str(e),
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        }), 500

@app.route("/conversation/<conversation_id>", methods=["GET"])
@log_api_call
def get_conversation(conversation_id: str):
    """Retrieve full conversation data."""
    logger.info(f"Retrieving conversation {conversation_id}")
    try:
        conversation = conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return jsonify({
                "type": "ERROR",
                "error": "Conversation not found"
            }), 404
            
        return jsonify({
            "type": "CONVERSATION",
            "conversation": conversation
        })
        
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        return jsonify({
            "type": "ERROR",
            "error": str(e)
        }), 500

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                       context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced metrics."""
    logger.info("Starting MATRIX processing")
    try:
        # Analyze patterns in current text and previous responses
        patterns = pattern_analyzer.analyze_patterns(current_text)
        logger.info(f"Pattern analysis results: {patterns}")
        
        # Get combined previous questions for similarity check
        previous_questions = [resp.get('question', '') for resp in previous_responses]
        similarity_scores = []
        
        for prev_q in previous_questions:
            if prev_q:
                similarity = pattern_analyzer.compute_similarity(current_text, prev_q)
                similarity_scores.append(similarity)
        
        # Calculate average similarity if we have previous questions
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        state = matrix.state_encoder.encode_state(
            [],
            previous_responses,
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
            previous_responses,
            current_text
        )
        
        # Enhanced logging for debugging
        logger.info(f"MATRIX metrics - Optimist: {optimist_view:.2f}, Pessimist: {pessimist_view:.2f}")
        logger.info(f"Average similarity score: {avg_similarity:.2f}")
        
        return {
            **matrix_output,
            "pattern_analysis": patterns,
            "similarity_score": avg_similarity,
            "optimist_score": optimist_view,
            "pessimist_score": pessimist_view
        }
        
    except Exception as e:
        logger.error(f"Error in MATRIX processing: {str(e)}")
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5},
            "pattern_analysis": {"optimist_confidence": 0.5, "pessimist_confidence": 0.5},
            "similarity_score": 0,
            "optimist_score": 0.5,
            "pessimist_score": 0.5
        }

# Update the should_move_to_examinations function with enhanced MATRIX metrics
def should_move_to_examinations(conversation: Dict) -> bool:
    """Determine if enough followup questions have been asked based on MATRIX metrics."""
    if len(conversation["followup_responses"]) >= MATRIXConfig.MAX_QUESTIONS:
        logger.info("Reached maximum number of questions")
        return True
        
    if not conversation["followup_responses"]:
        return False
        
    try:
        matrix_output = process_with_matrix(
            conversation["current_question"]["question"],
            conversation["followup_responses"]
        )
        
        # Enhanced MATRIX-based checks
        matrix_conditions = {
            "optimist_confident": matrix_output["optimist_score"] > 0.70,
            "pessimist_confident": matrix_output["pessimist_score"] > 0.70,
            "high_similarity": matrix_output["similarity_score"] > 0.85,
            "overall_confidence": matrix_output["confidence"] > 0.75
        }
        
        # Log decision factors
        logger.info("MATRIX decision factors:")
        for factor, value in matrix_conditions.items():
            logger.info(f"- {factor}: {value}")
        
        should_stop = (
            matrix_conditions["optimist_confident"] or
            matrix_conditions["pessimist_confident"] or
            matrix_conditions["high_similarity"] or
            matrix_conditions["overall_confidence"] or
            len(conversation["followup_responses"]) >= MATRIXConfig.MAX_FOLLOWUPS
        )
        
        if should_stop:
            reason = next(k for k, v in matrix_conditions.items() if v) if any(matrix_conditions.values()) else "max_followups"
            logger.info(f"Moving to examinations. Reason: {reason}")
            
        return should_stop
        
    except Exception as e:
        logger.error(f"Error in examination check: {str(e)}")
        return len(conversation["followup_responses"]) >= 3

@app.route("/reset", methods=["POST"])
@log_api_call
def reset_conversation():
    """Reset a conversation or clear all conversations."""
    try:
        data = request.json or {}
        conversation_id = data.get("conversationId")
        
        if conversation_id:
            logger.info(f"Resetting conversation {conversation_id}")
            if conversation_id in conversations:
                del conversations[conversation_id]
                return jsonify({
                    "type": "SUCCESS",
                    "message": "Conversation reset successfully"
                })
            else:
                return jsonify({
                    "type": "ERROR",
                    "error": "Conversation not found"
                }), 404
        else:
            logger.info("Clearing all conversations")
            conversations.clear()
            return jsonify({
                "type": "SUCCESS",
                "message": "All conversations cleared"
            })
            
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}")
        return jsonify({
            "type": "ERROR",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    logger.info("Starting Medical Assessment API")
    app.run(debug=True, host="0.0.0.0", port=5000)