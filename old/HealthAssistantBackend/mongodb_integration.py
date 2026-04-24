from typing import Dict, List, Any, Optional, Tuple
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
import json

# Load environment variables
load_dotenv()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "medical_assessment")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
sessions_collection = db["sessions"]
questions_collection = db["questions"]
examinations_collection = db["examinations"]

def create_or_update_session(chat_name: str) -> str:
    """
    Create a new session or return existing session ID.
    
    Parameters:
        chat_name (str): Name of the chat/session
        
    Returns:
        str: MongoDB ObjectId of the session as string
    """
    # Check if session exists
    session = sessions_collection.find_one({"chat_name": chat_name})
    
    if session:
        return str(session["_id"])
    
    # Create new session
    session_data = {
        "chat_name": chat_name,
        "initial_responses": [],
        "followup_responses": [],
        "exam_responses": [],
        "current_question_index": 0,
        "phase": "initial"
    }
    
    result = sessions_collection.insert_one(session_data)
    return str(result.inserted_id)

def get_session_data(chat_name: str) -> Dict[str, Any]:
    """
    Get session data for the given chat name.
    
    Parameters:
        chat_name (str): Name of the chat/session
        
    Returns:
        Dict[str, Any]: Session data
    """
    session = sessions_collection.find_one({"chat_name": chat_name})
    
    if not session:
        # Create new session if it doesn't exist
        session_id = create_or_update_session(chat_name)
        session = sessions_collection.find_one({"_id": ObjectId(session_id)})
    
    # Convert MongoDB _id to string for JSON serialization
    if session:
        session["_id"] = str(session["_id"])
    
    return session

def update_session_responses(chat_name: str, response_type: str, response_data: Dict[str, Any]) -> None:
    """
    Update session responses.
    
    Parameters:
        chat_name (str): Name of the chat/session
        response_type (str): Type of response (initial_responses, followup_responses, exam_responses)
        response_data (Dict[str, Any]): Response data to append
    """
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$push": {response_type: response_data}}
    )

def update_session_phase(chat_name: str, phase: str) -> None:
    """
    Update session phase.
    
    Parameters:
        chat_name (str): Name of the chat/session
        phase (str): New phase of the session
    """
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$set": {"phase": phase}}
    )

def store_question(chat_name: str, question_data: Dict[str, Any]) -> str:
    """
    Store a question in the questions collection.
    
    Parameters:
        chat_name (str): Name of the chat/session
        question_data (Dict[str, Any]): Question data
        
    Returns:
        str: MongoDB ObjectId of the inserted question as string
    """
    # Add chat_name to question data
    question_data["chat_name"] = chat_name
    
    result = questions_collection.insert_one(question_data)
    return str(result.inserted_id)

def get_structured_questions(chat_name: str) -> List[Dict[str, Any]]:
    """
    Get all structured questions for a chat session.
    
    Parameters:
        chat_name (str): Name of the chat/session
        
    Returns:
        List[Dict[str, Any]]: List of structured questions
    """
    questions = list(questions_collection.find({"chat_name": chat_name}))
    
    # Convert MongoDB _id to string for JSON serialization
    for question in questions:
        question["_id"] = str(question["_id"])
    
    return questions

def store_examination(chat_name: str, examination_data: Dict[str, Any]) -> str:
    """
    Store examination data in the examinations collection.
    
    Parameters:
        chat_name (str): Name of the chat/session
        examination_data (Dict[str, Any]): Examination data
        
    Returns:
        str: MongoDB ObjectId of the inserted examination as string
    """
    # Add chat_name to examination data
    examination_data["chat_name"] = chat_name
    
    result = examinations_collection.insert_one(examination_data)
    return str(result.inserted_id)

def get_examination_history(chat_name: str) -> List[Dict[str, Any]]:
    """
    Get examination history for a chat session.
    
    Parameters:
        chat_name (str): Name of the chat/session
        
    Returns:
        List[Dict[str, Any]]: List of examinations
    """
    examinations = list(examinations_collection.find({"chat_name": chat_name}))
    
    # Convert MongoDB _id to string for JSON serialization
    for examination in examinations:
        examination["_id"] = str(examination["_id"])
    
    return examinations

def extract_question_and_options(chat_name: str, question_number: int) -> List[Any]:
    """
    Extracts the question text and its options for a specific question number.
    
    Parameters:
        chat_name (str): Name of the chat/session
        question_number (int): The question number (1-indexed)
    
    Returns:
        List[Any]: An array with the question text as the first element followed
                  by each option. Returns an empty list if the question number is
                  out of range.
    """
    questions = get_structured_questions(chat_name)
    
    # Calculate the index (question_number is assumed to be 1-indexed)
    index = question_number - 1

    if index < 0 or index >= len(questions):
        print(f"Question number {question_number} is out of range.")
        return []

    # Retrieve the corresponding question map.
    question_map = questions[index]
    questionVals = []

    # Extract and print the question text.
    question_text = question_map.get("question", "")
    print(f"Question {question_number}: {question_text}")
    questionVals.append(question_text)

    # Extract and print each option.
    options = question_map.get("options", [])
    for option in options:
        print("Option:", option)
        questionVals.append(option)

    return questionVals

def extract_examination_and_options(chat_name: str, exam_number: int) -> List[Any]:
    """
    Extracts the examination text and its options for a specific examination number.
    
    Parameters:
        chat_name (str): Name of the chat/session
        exam_number (int): The exam number (1-indexed)
    
    Returns:
        List[Any]: An array with the examination text as the first element followed
                  by each option. Returns an empty list if the exam number is out of range.
    """
    examinations = get_examination_history(chat_name)
    
    # Calculate the index (exam_number is assumed to be 1-indexed)
    index = exam_number - 1

    if index < 0 or index >= len(examinations):
        print(f"Exam number {exam_number} is out of range.")
        return []

    # Retrieve the corresponding examination map.
    exam_map = examinations[index]
    examVals = []

    # Extract and print the examination text.
    exam_text = exam_map.get("examination", "")
    print(f"Examination {exam_number}: {exam_text}")
    examVals.append(exam_text)

    # Extract and print each option.
    options = exam_map.get("options", [])
    for option in options:
        print("Option:", option)
        examVals.append(option)

    return examVals

def parse_question_data(chat_name: str, question: str, options: list, answer: str, 
                       matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format and store it in MongoDB."""
    
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
    
    # Store in MongoDB
    question_id = store_question(chat_name, question_data)
    question_data["_id"] = question_id
    
    return question_data

def parse_examination_text(examination_text: str) -> tuple[str, list[str]]:
    """
    Parse examination text using "#:" delimiter.
    Returns tuple of (examination text, list of options).
    """
    # Split on "#:" delimiter
    parts = examination_text.split("#:")
    
    if len(parts) < 2:
        raise ValueError("Invalid examination format - missing '#:' delimiter")
        
    # First part is the examination text
    examination = parts[0].strip()
    
    # Remaining parts are options
    options = [opt.strip() for opt in parts[1:] if opt.strip()]
    
    return examination, options

def store_examination_data(chat_name: str, examination_text: str, selected_option: int):
    """
    Store examination data in MongoDB.
    
    Parameters:
        chat_name (str): Name of the chat/session
        examination_text (str): Text of the examination
        selected_option (int): Selected option index
    """
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
        
        # Store in MongoDB
        store_examination(chat_name, examination_entry)
        
    except Exception as e:
        print(f"Error storing examination: {e}")

def print_mongo_collections(chat_name: str):
    """Print both MongoDB collections with proper formatting."""
    
    questions = get_structured_questions(chat_name)
    examinations = get_examination_history(chat_name)
    
    print("\nCurrent Structured Questions Array:")
    print("================================")
    print(json.dumps(questions, indent=2))
    
    print("\nCurrent Examination History:")
    print("=========================")
    print(json.dumps(examinations, indent=2))