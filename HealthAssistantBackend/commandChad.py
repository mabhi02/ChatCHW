from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import json
from pymongo import MongoClient
from bson.objectid import ObjectId

# Import MATRIX components
from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

# Load environment variables
load_dotenv()

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

def create_or_get_session(chat_name: str) -> Dict[str, Any]:
    """Create a new session or get an existing session for the given chat name."""
    # Check if session exists
    session = sessions_collection.find_one({"chat_name": chat_name})
    
    if not session:
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
        session = sessions_collection.find_one({"_id": result.inserted_id})
    
    # Convert MongoDB _id to string for easier handling
    if session:
        session["_id"] = str(session["_id"])
    
    return session

def update_session_responses(chat_name: str, response_type: str, response_data: Dict[str, Any]) -> None:
    """Update session responses."""
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$push": {response_type: response_data}}
    )

def update_session_phase(chat_name: str, phase: str) -> None:
    """Update session phase."""
    sessions_collection.update_one(
        {"chat_name": chat_name},
        {"$set": {"phase": phase}}
    )

def store_question(chat_name: str, question_data: Dict[str, Any]) -> str:
    """Store a question in the questions collection."""
    # Add chat_name to question data
    question_data["chat_name"] = chat_name
    
    result = questions_collection.insert_one(question_data)
    return str(result.inserted_id)

def store_examination(chat_name: str, examination_data: Dict[str, Any]) -> str:
    """Store examination data in the examinations collection."""
    # Add chat_name to examination data
    examination_data["chat_name"] = chat_name
    
    result = examinations_collection.insert_one(examination_data)
    return str(result.inserted_id)

def get_structured_questions(chat_name: str) -> List[Dict[str, Any]]:
    """Get all structured questions for a chat session."""
    questions = list(questions_collection.find({"chat_name": chat_name}))
    
    # Convert MongoDB _id to string for JSON serialization
    for question in questions:
        question["_id"] = str(question["_id"])
    
    return questions

def get_examination_history(chat_name: str) -> List[Dict[str, Any]]:
    """Get examination history for a chat session."""
    examinations = list(examinations_collection.find({"chat_name": chat_name}))
    
    # Convert MongoDB _id to string for JSON serialization
    for examination in examinations:
        examination["_id"] = str(examination["_id"])
    
    return examinations

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
            options.append(line)
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

def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> bool:
    """Judge if the current question is too similar using MATRIX."""
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

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> bool:
    """Judge examination similarity with improved duplicate detection."""
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

def parse_question_data(chat_name: str, question: str, options: list, answer: str, 
                       matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format and store in MongoDB."""
    
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
    
    return question_data

def store_examination_data(chat_name: str, examination_text: str, selected_option: int):
    """Parse examination text and store in MongoDB."""
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

def print_options(options: List[Dict[str, Any]]) -> None:
    """Print formatted options for multiple choice questions."""
    for option in options:
        print(f"  {option['id']}: {option['text']}")

def validate_num_input(value: str, range_data: Dict[str, int]) -> Optional[int]:
    """Validate numeric input against given range."""
    try:
        num = int(value)
        if range_data['min'] <= num <= range_data['max']:
            return num
        return None
    except ValueError:
        return None

def validate_mc_input(value: str, options: List[Dict[str, Any]]) -> Optional[str]:
    """Validate multiple choice input against given options."""
    valid_ids = [str(opt['id']) for opt in options]
    return value if value in valid_ids else None

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              followup_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate diagnosis and treatment recommendations."""
    try:
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get only the most relevant findings for context
        key_findings = []
        for resp in followup_responses + exam_responses:
            if isinstance(resp.get('answer'), str):
                key_findings.append(f"{resp['answer']}")
        key_findings = key_findings[-3:]  # Only keep last 3 findings
        
        # First, get diagnosis using minimal context
        index = pc.Index("final-asha")
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

def print_mongo_collections(chat_name: str):
    """Print MongoDB collections for the chat session."""
    
    questions = get_structured_questions(chat_name)
    examinations = get_examination_history(chat_name)
    
    print("\nCurrent Structured Questions (MongoDB):")
    print("====================================")
    print(json.dumps(questions, indent=2))
    
    print("\nCurrent Examination History (MongoDB):")
    print("===================================")
    print(json.dumps(examinations, indent=2))

def extract_question_and_options(chat_name: str, question_number: int) -> List[Any]:
    """Extract question and options from MongoDB."""
    questions = get_structured_questions(chat_name)
    
    # Calculate the index (question_number is assumed to be 1-indexed)
    index = question_number - 1

    if index < 0 or index >= len(questions):
        print(f"Question number {question_number} is out of range.")
        return []

    # Retrieve the corresponding question map
    question_map = questions[index]
    questionVals = []

    # Extract and print the question text
    question_text = question_map.get("question", "")
    print(f"Question {question_number}: {question_text}")
    questionVals.append(question_text)

    # Extract and print each option
    options = question_map.get("options", [])
    for option in options:
        print("Option:", option)
        questionVals.append(option)

    return questionVals

def extract_examination_and_options(chat_name: str, exam_number: int) -> List[Any]:
    """Extract examination and options from MongoDB."""
    examinations = get_examination_history(chat_name)
    
    # Calculate the index (exam_number is assumed to be 1-indexed)
    index = exam_number - 1

    if index < 0 or index >= len(examinations):
        print(f"Exam number {exam_number} is out of range.")
        return []

    # Retrieve the corresponding examination map
    exam_map = examinations[index]
    examVals = []

    # Extract and print the examination text
    exam_text = exam_map.get("examination", "")
    print(f"Examination {exam_number}: {exam_text}")
    examVals.append(exam_text)

    # Extract and print each option
    options = exam_map.get("options", [])
    for option in options:
        print("Option:", option)
        examVals.append(option)

    return examVals

def main():
    try:
        # Get chat name for MongoDB session
        chat_name = input("Enter patient chat/session name: ").strip()
        if not chat_name:
            chat_name = "default_session"
            print(f"Using default session name: {chat_name}")
        
        # Initialize or get session data
        session = create_or_get_session(chat_name)
        print(f"\nSession: {chat_name} (ID: {session['_id']})")
        
        initial_responses = session.get("initial_responses", [])
        followup_responses = session.get("followup_responses", [])
        exam_responses = session.get("exam_responses", [])
        
        # Skip initial questions if already answered
        if initial_responses:
            print("\nFound existing initial responses:")
            for resp in initial_responses:
                print(f"Q: {resp['question']}")
                print(f"A: {resp['answer']}\n")
            proceed = input("Continue with these responses? (y/n): ").strip().lower()
            if proceed != 'y':
                initial_responses = []
                # Update session to clear initial responses
                sessions_collection.update_one(
                    {"chat_name": chat_name},
                    {"$set": {"initial_responses": []}}
                )
        
        # Initial Questions Loop (if needed)
        if not initial_responses:
            print("\nMedical Assessment Initial Questions")
            print("===================================")
            
            for question in questions_init:
                while True:
                    print(f"\n{question['question']}")
                    
                    if question['type'] in ['MC', 'YN', 'MCM']:
                        print_options(question['options'])
                        answer = input("Enter your choice (enter the number or id): ").strip()
                        
                        if question['type'] == 'MCM':
                            print("For multiple selections, separate with commas (e.g., 1,2,3)")
                            if ',' in answer:
                                answers = answer.split(',')
                                valid = all(validate_mc_input(a.strip(), question['options']) for a in answers)
                                if valid:
                                    response_data = {
                                        "question": question['question'],
                                        "answer": [a.strip() for a in answers],
                                        "type": question['type']
                                    }
                                    initial_responses.append(response_data)
                                    update_session_responses(chat_name, "initial_responses", response_data)
                                    break
                            else:
                                if validate_mc_input(answer, question['options']):
                                    response_data = {
                                        "question": question['question'],
                                        "answer": [answer],
                                        "type": question['type']
                                    }
                                    initial_responses.append(response_data)
                                    update_session_responses(chat_name, "initial_responses", response_data)
                                    break
                        else:
                            if validate_mc_input(answer, question['options']):
                                if answer == "5":
                                    custom_answer = input("Please specify: ").strip()
                                    response_data = {
                                        "question": question['question'],
                                        "answer": custom_answer,
                                        "type": question['type']
                                    }
                                    initial_responses.append(response_data)
                                    update_session_responses(chat_name, "initial_responses", response_data)
                                else:
                                    selected_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                                    response_data = {
                                        "question": question['question'],
                                        "answer": selected_text,
                                        "type": question['type']
                                    }
                                    initial_responses.append(response_data)
                                    update_session_responses(chat_name, "initial_responses", response_data)
                                break
                    
                    elif question['type'] == 'NUM':
                        answer = input(f"Enter a number between {question['range']['min']} and {question['range']['max']}: ")
                        if validated_num := validate_num_input(answer, question['range']):
                            response_data = {
                                "question": question['question'],
                                "answer": validated_num,
                                "type": question['type']
                            }
                            initial_responses.append(response_data)
                            update_session_responses(chat_name, "initial_responses", response_data)
                            break
                            
                    elif question['type'] == 'FREE':
                        answer = input("Enter your response (type your answer and press Enter): ").strip()
                        if answer:
                            response_data = {
                                "question": question['question'],
                                "answer": answer,
                                "type": question['type']
                            }
                            initial_responses.append(response_data)
                            update_session_responses(chat_name, "initial_responses", response_data)
                            break
                    
                    print("Invalid input, please try again.")

            print("\nThank you for providing your information. Here's what we recorded:\n")
            for resp in initial_responses:
                print(f"Q: {resp['question']}")
                print(f"A: {resp['answer']}\n")
            
            # Update session phase
            update_session_phase(chat_name, "followup")

        # Skip follow-up questions if already answered and user wants to keep them
        if followup_responses:
            print("\nFound existing follow-up responses:")
            for i, resp in enumerate(followup_responses, 1):
                print(f"{i}. Q: {resp['question']}")
                print(f"   A: {resp['answer']}\n")
            proceed = input("Continue with these responses? (y/n): ").strip().lower()
            if proceed != 'y':
                followup_responses = []
                # Update session to clear followup responses
                sessions_collection.update_one(
                    {"chat_name": chat_name},
                    {"$set": {"followup_responses": []}}
                )
                # Clear existing questions in MongoDB for this chat
                questions_collection.delete_many({"chat_name": chat_name})

        
        # Replace the examination loop section with this code:

        # Examinations Loop (if needed)
        if not exam_responses:
            # Get the initial complaint
            initial_complaint = next((resp['answer'] for resp in initial_responses 
                                if resp['question'] == "Please describe what brings you here today"), "")
            
            print("\nBased on your responses, I'll recommend appropriate examinations.")
            index = pc.Index("final-asha")
            exam_citations = []
            
            # Track symptoms
            symptoms = set()
            for resp in followup_responses:
                answer = resp['answer'].lower() if isinstance(resp['answer'], str) else ""
                for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                            'headache', 'nausea', 'dizziness', 'rash']:
                    if symptom in answer:
                        symptoms.add(symptom)
            
            while True:
                try:
                    # Compress context for examination recommendations
                    context = f"""Initial complaint: {initial_complaint}
        Key symptoms: {', '.join(symptoms)}
        Previous findings: {str([exam.get('examination', '') for exam in exam_responses]) if exam_responses else "None"}"""

                    embedding = get_embedding_batch([context])[0]
                    relevant_docs = vectorQuotesWithSource(embedding, index)
                    
                    if not relevant_docs:
                        print("Error: Could not generate relevant examination.")
                        continue
                    
                    exam_citations.extend(relevant_docs)
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
                    
                    if judge_exam(exam_responses, examination_text):
                        # Update session phase
                        update_session_phase(chat_name, "diagnosis")
                        break
                        
                    try:
                        # Print the raw examination text for debugging
                        print("\nExamination text received from AI:")
                        print(examination_text)
                        
                        # Parse the examination text with improved function
                        examination, option_texts = parse_examination_text(examination_text)
                        
                        # Create options list
                        options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
                        options.append({"id": 5, "text": "Other (please specify)"})
                        
                        print(f"\nRecommended Examination:")
                        print(examination)
                        print("\nSelect the finding:")
                        print_options(options)
                        
                        while True:
                            answer = input("Enter the finding (enter the number): ").strip()
                            
                            if validate_mc_input(answer, options):
                                # Store the examination with its parsed format in MongoDB
                                # Even if the original text had no "#:" delimiters
                                formatted_text = examination + "\n\n"
                                for opt in option_texts:
                                    formatted_text += f"#:{opt}\n"
                                    
                                store_examination_data(chat_name, formatted_text, int(answer))
                                
                                if answer == "5":
                                    custom_result = input("Please specify the finding: ").strip()
                                    response_data = {
                                        "examination": examination,
                                        "result": custom_result,
                                        "type": "EXAM",
                                        "citations": exam_citations[-5:]
                                    }
                                    exam_responses.append(response_data)
                                    update_session_responses(chat_name, "exam_responses", response_data)
                                else:
                                    selected_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                                    response_data = {
                                        "examination": examination,
                                        "result": selected_text,
                                        "type": "EXAM",
                                        "citations": exam_citations[-5:]
                                    }
                                    exam_responses.append(response_data)
                                    update_session_responses(chat_name, "exam_responses", response_data)
                                
                                # Print MongoDB collections
                                print_mongo_collections(chat_name)
                                break
                                
                            print("Invalid input, please try again.")
                            
                    except ValueError as e:
                        print(f"Error parsing examination: {e}")
                        print("Trying again with a different format...")
                        continue
                        
                except Exception as e:
                    print(f"Error generating examination: {e}")
                    continue


        
        # Generate diagnosis and treatment
        print("\nGenerating diagnosis and treatment recommendations...")
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
        
        # Print diagnosis
        print("\nDiagnosis:")
        print("==========")
        print(results["diagnosis"])
        
        # Print treatment plan
        print("\nRecommended Treatment Plan:")
        print("=========================")
        print(results["treatment"])
        
        # Print references
        print("\nKey References:")
        print("==============")
        seen_sources = set()
        for citation in results["citations"]:
            if citation['source'] not in seen_sources:
                print(f"- {citation['source']} (relevance: {citation['score']:.2f})")
                seen_sources.add(citation['source'])
        
        # Print questions and examinations from MongoDB
        print("\nFinal Structured Questions (MongoDB):")
        print("==================================")
        questions = get_structured_questions(chat_name)
        print(json.dumps(questions, indent=2))
        
        if questions:
            extract_question_and_options(chat_name, 1)
        
        print("\nFinal Examination History (MongoDB):")
        print("================================")
        examinations = get_examination_history(chat_name)
        print(json.dumps(examinations, indent=2))
        
        if examinations:
            extract_examination_and_options(chat_name, 1)
        
        print("\nAssessment complete! Data stored in MongoDB.")
        print(f"Session: {chat_name} (ID: {session['_id']})")
        
    except KeyboardInterrupt:
        print("\nAssessment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()