from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import torch
import torch.nn as nn
import json

from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig
from prompts_infer import (
    get_diagnosis_prompt,
    get_treatment_prompt,
    get_main_followup_question_prompt,
    get_main_examination_prompt
)


examination_history = []
structured_questions_array = []


# Load environment variables and initialize clients
load_dotenv()
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


def get_openai_completion(prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
    """
    Get completion from OpenAI's GPT-4 API.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",  # Using GPT-4-mini
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


def generate_question_with_options(input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a follow-up question with options based on patient input.
    Uses predefined templates to avoid API rate limits.
    """
    try:
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in input_data 
                            if resp['question'] == "Please describe what brings you here today"), "").lower()

        # Common question patterns based on symptoms
        pain_questions = [
            {
                "question": "How long have you been experiencing this pain?",
                "options": [
                    {"id": 1, "text": "Less than 24 hours"},
                    {"id": 2, "text": "1-7 days"},
                    {"id": 3, "text": "1-4 weeks"},
                    {"id": 4, "text": "More than a month"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            },
            {
                "question": "How would you describe the severity of the pain?",
                "options": [
                    {"id": 1, "text": "Mild - noticeable but not interfering with activities"},
                    {"id": 2, "text": "Moderate - somewhat interfering with activities"},
                    {"id": 3, "text": "Severe - significantly interfering with activities"},
                    {"id": 4, "text": "Very severe - unable to perform activities"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            },
            {
                "question": "What makes the pain worse?",
                "options": [
                    {"id": 1, "text": "Movement or physical activity"},
                    {"id": 2, "text": "Pressure or touch"},
                    {"id": 3, "text": "Specific positions"},
                    {"id": 4, "text": "Nothing specific"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            }
        ]

        general_questions = [
            {
                "question": "When did your symptoms first begin?",
                "options": [
                    {"id": 1, "text": "Within the last 24 hours"},
                    {"id": 2, "text": "In the past week"},
                    {"id": 3, "text": "Several weeks ago"},
                    {"id": 4, "text": "More than a month ago"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            },
            {
                "question": "How often do you experience these symptoms?",
                "options": [
                    {"id": 1, "text": "Constantly"},
                    {"id": 2, "text": "Several times a day"},
                    {"id": 3, "text": "A few times a week"},
                    {"id": 4, "text": "Occasionally"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            },
            {
                "question": "How does this affect your daily activities?",
                "options": [
                    {"id": 1, "text": "Not at all"},
                    {"id": 2, "text": "Slightly limiting"},
                    {"id": 3, "text": "Moderately limiting"},
                    {"id": 4, "text": "Severely limiting"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
            }
        ]

        # Determine which question set to use
        if "pain" in initial_complaint:
            questions = pain_questions
        else:
            questions = general_questions

        # Get previous questions
        asked_questions = set(resp.get('question', '') for resp in input_data if resp.get('question'))

        # Find first unused question
        for question_data in questions:
            if question_data['question'] not in asked_questions:
                return {
                    "question": question_data['question'],
                    "options": question_data['options'],
                    "type": "MC"
                }

        # If all questions used, return a general follow-up
        return {
            "question": "Are you experiencing any other symptoms?",
            "options": [
                {"id": 1, "text": "No other symptoms"},
                {"id": 2, "text": "Yes, mild additional symptoms"},
                {"id": 3, "text": "Yes, moderate additional symptoms"},
                {"id": 4, "text": "Yes, severe additional symptoms"},
                {"id": 5, "text": "Other (please specify)"}
            ],
            "type": "MC"
        }

    except Exception as e:
        print(f"Error generating question: {e}")
        # Return a fallback question if there's an error
        return {
            "question": "How long have you been experiencing these symptoms?",
            "options": [
                {"id": 1, "text": "Less than 24 hours"},
                {"id": 2, "text": "1-7 days"},
                {"id": 3, "text": "1-4 weeks"},
                {"id": 4, "text": "More than a month"},
                {"id": 5, "text": "Other (please specify)"}
            ],
            "type": "MC"
        }

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                       context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced error handling."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        print(f"Pattern Analysis - Optimist: {patterns['optimist_confidence']:.2f}, "
              f"Pessimist: {patterns['pessimist_confidence']:.2f}")
        
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
        
        return matrix_output
        
    except Exception as e:
        print(f"Warning: MATRIX processing error: {e}")
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5}
        }

# Add these at the top of your existing chad.py, after the imports but before questions_init

# Initialize global arrays
examination_history = []
structured_questions_array = []

def initialize_session():
    """Initialize/reset the global arrays for a new session."""
    global structured_questions_array, examination_history
    structured_questions_array = []
    examination_history = []
    return {
        "initial_responses": [],
        "followup_responses": [],
        "exam_responses": [],
        "structured_questions": [],
        "examinations": [],
        "current_question_index": 0,
        "phase": "initial"
    }

def get_session_data(session_id, sessions_dict):
    """Get or create session data for the given session ID."""
    if session_id not in sessions_dict:
        sessions_dict[session_id] = initialize_session()
    return sessions_dict[session_id]

# Modified store_examination function to handle sessions
def store_examination(examination_text: str, selected_option: int):
    """Store examination data in the global examination history."""
    global examination_history
    
    try:
        # Parse examination and options
        examination, options = parse_examination_text(examination_text)
        
        # Create examination entry
        examination_entry = {
            "examination": examination,
            "options": options,
            "selected_option": selected_option
        }
        
        examination_history.append(examination_entry)
        
    except Exception as e:
        print(f"Error storing examination: {e}")

# Modified judge function to handle sessions
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

# Add this to the end of your chad.py file
__all__ = [
    'questions_init',
    'structured_questions_array',
    'examination_history',
    'get_embedding_batch',
    'vectorQuotesWithSource',
    'process_with_matrix',
    'judge',
    'judge_exam',
    'parse_examination_text',
    'get_diagnosis_and_treatment',
    'parse_question_data',
    'store_examination',
    'generate_question_with_options',
    'initialize_session',
    'get_session_data'
]

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

structured_questions_array = []

def store_question(question_data: dict):
    """Store a question in the global array."""
    global structured_questions_array
    structured_questions_array.append(question_data)

def parse_question_data(question: str, options: list, answer: str, matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format."""
    global structured_questions_array
    
    # Convert option list to formatted strings
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    
    # Get selected option index (1-based indexing)
    selected_idx = next((i+1 for i, opt in enumerate(options) 
                        if opt['text'] == answer), None)
    
    # Format sources from citations
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" 
              for cite in citations] if citations else []
    
    question_data = {
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
    
    structured_questions_array.append(question_data)
    return question_data


def parse_examination_text(text):
    """
    Parse examination text to extract the examination description and findings.
    Handles three scenarios:
    1. Clear examination procedure with findings
    2. No examination information available
    3. Partial/informal examination information
    
    Args:
        text (str): The examination text to parse
    
    Returns:
        tuple: (examination_description, list_of_findings)
    """
    # Process empty or very short responses
    if not text or len(text.strip()) < 10:
        examination = "No examination information provided in the medical guide."
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    # Common phrases indicating no information is available
    no_info_phrases = [
        "does not provide", "no information", "no examination", 
        "no specific examination", "doesn't provide", "isn't provided",
        "the guide doesn't provide", "not available in the guide",
        "not included in the guide", "not mentioned in the guide",
        "doesn't contain", "does not mention"
    ]
    
    # SCENARIO 1: Check if the response indicates no examination information is available
    if any(phrase in text.lower() for phrase in no_info_phrases):
        # Return a standardized "No examination" response with default findings
        examination = text.strip()
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    # SCENARIO 2: Check if there's some useful information but not a formal procedure
    if ("relevant information" in text.lower() or "some information" in text.lower() or 
        "general guidance" in text.lower() or "may be helpful" in text.lower() or
        not any(line.startswith('#') or line.startswith('#:') for line in text.strip().split('\n'))):
        
        # Extract the useful information
        examination = "PARTIAL INFORMATION (Not a formal examination procedure):\n" + text.strip()
        findings = [
            "Consider referring to medical professional for proper examination",
            "Use this information as supplementary guidance only",
            "Document observations based on general assessment",
            "Consult with supervisor about next steps"
        ]
        return examination, findings
    
    # SCENARIO 3: Regular parsing for normal examination text with findings
    # Split the text by lines
    lines = text.strip().split('\n')
    
    # Extract examination description (everything before the first finding)
    examination_lines = []
    findings = []
    
    in_examination = True
    
    for line in lines:
        line = line.strip()
        # Check for both '#:' and '#' formats
        if line.startswith('#:') or (line.startswith('#') and not line.startswith('#:')):
            in_examination = False
            # Extract the finding text by removing the delimiter
            if line.startswith('#:'):
                finding = line[2:].strip()
            else:
                finding = line[1:].strip()
            findings.append(finding)
        elif in_examination and line:
            examination_lines.append(line)
    
    # Join the examination lines
    examination = '\n'.join(examination_lines)
    
    # If we didn't find any findings but have examination text, create default findings
    if not findings and examination:
        findings = [
            "Normal finding",
            "Abnormal finding requiring further assessment",
            "Inconclusive finding - may need additional tests",
            "Unable to determine based on current examination"
        ]
    # If we don't have examination text or findings, provide a fallback response
    elif not findings and not examination:
        examination = "The response from the medical guide was incomplete or invalid."
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
    
    return examination, findings

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              followup_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get diagnosis and treatment based on responses"""
    try:
        # Get initial complaint
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get symptoms from followup responses
        symptoms = []
        for resp in followup_responses:
            if isinstance(resp.get('answer'), str) and not resp.get('answer').startswith("No "):
                symptoms.append(resp.get('answer'))
        
        # Format symptoms text
        symptoms_text = "\n".join([f"- {symptom}" for symptom in symptoms])
        
        # Format examination results
        exam_results = []
        for exam in exam_responses:
            if 'examination' in exam and 'result' in exam:
                exam_results.append(f"Examination: {exam['examination']}\nResult: {exam['result']}")
        exam_results_text = "\n\n".join(exam_results)
        
        # Get embeddings for context
        context_items = [initial_complaint]
        context_items.extend(symptoms)
        context_items.extend(exam_results)
        
        embeddings = get_embedding_batch(context_items)
        
        # Get relevant documents using embeddings
        index = pc.Index("who-guide-old")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=2)
            if matches:
                relevant_matches.extend(matches)
        
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
        
        # Format medical guide content
        medical_guide_content = "\n\n".join([match['text'] for match in top_matches])
        
        # Get diagnosis
        diagnosis_prompt = get_diagnosis_prompt(
            initial_complaint=initial_complaint,
            symptoms_text=symptoms_text,
            exam_results_text=exam_results_text,
            medical_guide_content=medical_guide_content
        )
        
        diagnosis_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": diagnosis_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        
        # Get treatment
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
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": treatment_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        treatment = treatment_completion.choices[0].message.content.strip()
        
        # Format citations
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

def print_global_arrays():
    """Print both global arrays with proper formatting."""
    print("\nCurrent Structured Questions Array:")
    print("================================")
    print(json.dumps(structured_questions_array, indent=2))
    
    print("\nCurrent Examination History:")
    print("=========================")
    print(json.dumps(examination_history, indent=2))

def extract_question_and_options(questions: List[Dict[str, Any]], question_number: int) -> List[Any]:
    """
    Extracts the question text and its options from the list of questions
    using the provided question number. Prints them out and returns all the
    values as an array called questionVals.
    
    Parameters:
        questions (List[Dict[str, Any]]): A list of dictionaries, where each
                                          dictionary represents a question.
        question_number (int): The question number (1-indexed). For example, if
                               question_number is 1, the function will process
                               questions[0].
    
    Returns:
        List[Any]: An array with the question text as the first element followed
                   by each option. Returns an empty list if the question number is
                   out of range.
    """
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

def extract_examination_and_options(examinations: List[Dict[str, Any]], exam_number: int) -> List[Any]:
    """
    Extracts the examination text and its options from the list of examinations
    using the provided exam number. Prints them out and returns all the values
    as an array called examVals.
    
    Parameters:
        examinations (List[Dict[str, Any]]): A list of dictionaries, where each
                                              dictionary represents an examination.
        exam_number (int): The exam number (1-indexed). For example, if exam_number
                           is 1, the function will process examinations[0].
    
    Returns:
        List[Any]: An array with the examination text as the first element followed
                   by each option. Returns an empty list if the exam number is out of range.
    """
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

def main():
    """Main function to run the CHW assistant"""
    try:
        # Initialize session
        session_data = initialize_session()
        initial_responses = []
        followup_responses = []
        exam_responses = []
        
        # Initial Questions Loop
        print("\nWelcome to the Community Health Worker Assistant.")
        print("I'll ask you some initial questions to understand your situation.")
        
        for question in questions_init:
            print(f"\n{question['question']}")
            
            if question['type'] in ['MC', 'MCM']:
                print_options(question['options'])
                
                while True:
                    answer = input("Enter your choice (enter the number): ").strip()
                    
                    if validate_mc_input(answer, question['options']):
                        if answer == "5":  # "Other" option
                            custom_answer = input("Please specify your answer: ").strip()
                            answer_text = custom_answer
                        else:
                            answer_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                        
                        initial_responses.append({
                            "question": question['question'],
                            "answer": answer_text,
                            "type": question['type']
                        })
                        break
                
                print("Invalid input, please try again.")

        print("\nThank you for providing your information. Here's what we recorded:\n")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")

        # Follow-up Questions Loop
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        print("\nBased on your responses, I'll ask some follow-up questions.")
        index = pc.Index("who-guide-old")
        
        while True:
            try:
                context = f"Initial complaint: {initial_complaint}\n"
                if followup_responses:
                    context += "Previous responses:\n"
                    for resp in followup_responses:
                        context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
                
                embedding = get_embedding_batch([context])[0]
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                if not relevant_docs:
                    print("Error: Could not generate relevant question.")
                    continue
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
                
                prompt = get_main_followup_question_prompt(
                    initial_complaint=initial_complaint,
                    previous_questions=previous_questions,
                    combined_context=combined_context
                )
                
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
                
                # Use MATRIX to judge similarity and get pattern analysis
                matrix_output = process_with_matrix(question, followup_responses)
                
                # Print question and get answer
                print(f"\n{question}")
                print_options(options)
                
                while True:
                    answer = input("Enter your choice (enter the number): ").strip()
                    
                    if validate_mc_input(answer, options):
                        if answer == "5":
                            custom_answer = input("Please specify your answer: ").strip()
                            answer_text = custom_answer
                        else:
                            answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                        
                        # Store response
                        followup_responses.append({
                            "question": question,
                            "answer": answer_text,
                            "type": "MC",
                            "citations": relevant_docs[-5:]
                        })
                        
                        # Parse and store question data
                        parse_question_data(
                            question, 
                            options, 
                            answer_text, 
                            matrix_output, 
                            relevant_docs[-5:]
                        )
                        
                        # Print global arrays after each answer
                        print("\nCurrent Structured Questions Array:")
                        print("================================")
                        print(json.dumps(structured_questions_array, indent=2))
                        
                        print("\nCurrent Examination History:")
                        print("=========================")
                        print(json.dumps(examination_history, indent=2))
                        break
                        
                    print("Invalid input, please try again.")
                
                # Check if we should stop asking questions
                if judge(followup_responses, question):
                    print("\nSufficient information gathered. Moving to next phase...")
                    break
                    
            except Exception as e:
                print(f"Error generating follow-up question: {e}")
                continue

        # Examinations Loop
        print("\nBased on your responses, I'll recommend appropriate examinations.")
        exam_citations = []
        
        # Track symptoms
        symptoms = set()
        for resp in followup_responses:
            answer = resp['answer'].lower()
            for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                           'headache', 'nausea', 'dizziness', 'rash']:
                if symptom in answer:
                    symptoms.add(symptom)
        
        while True:
            try:
                # Compress context for examination recommendations
                context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}
Previous findings: {str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"}"""

                embedding = get_embedding_batch([context])[0]
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                if not relevant_docs:
                    print("Error: Could not generate relevant examination.")
                    continue
                
                exam_citations.extend(relevant_docs)
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                prompt = get_main_examination_prompt(
                    initial_complaint=initial_complaint,
                    symptoms=', '.join(symptoms),
                    previous_exams=str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"
                )

                examination_text = get_openai_completion(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.3
                )
                
                if judge_exam(exam_responses, examination_text):
                    break
                    
                try:
                    # Parse the examination text
                    examination, option_texts = parse_examination_text(examination_text)
                    
                    # Create options list
                    options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
                    options.append({"id": 5, "text": "Other (please specify)"})
                    
                    print(f"\nRecommended Examination:")
                    print(examination)
                    print("\nSelect the finding:")
                    print_options(options)
                    
                    while True:
                        answer = input("Enter your choice (enter the number): ").strip()
                        
                        if validate_mc_input(answer, options):
                            if answer == "5":
                                custom_answer = input("Please specify your answer: ").strip()
                                answer_text = custom_answer
                            else:
                                answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                            
                            # Store examination response
                            exam_responses.append({
                                "examination": examination,
                                "result": answer_text,
                                "type": "MC",
                                "citations": relevant_docs[-5:]
                            })
                            
                            # Store examination
                            store_examination(examination_text, int(answer))
                            break
                            
                        print("Invalid input, please try again.")
                        
                except Exception as e:
                    print(f"Error parsing examination: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error generating examination: {e}")
                continue

        # Get diagnosis and treatment
        results = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
        
        print("\nAssessment Complete")
        print("==================")
        print("\nDiagnosis:")
        print(results['diagnosis'])
        print("\nTreatment Plan:")
        print(results['treatment'])
        
        if results['citations']:
            print("\nKey References:")
            for cite in results['citations']:
                print(f"- {cite['source']} (relevance: {cite['score']:.2f})")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()