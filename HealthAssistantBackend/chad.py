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
        index = pc.Index("who-guide-old")
        diagnosis_embedding = get_embedding_batch([initial_complaint + " diagnosis"])[0]
        diagnosis_docs = vectorQuotesWithSource(diagnosis_embedding, index, top_k=3)  # Increased to get more context
        
        if not diagnosis_docs:
            return {
                "diagnosis": "Unable to provide diagnosis. No relevant information found in the medical guide. Please refer the patient to the nearest health facility.",
                "treatment": "Unable to provide treatment recommendations. No relevant information found in the medical guide. Please refer the patient to the nearest health facility.",
                "citations": [],
                "chunks_used": []
            }
        
        # Save all chunks to return them
        all_chunks_used = []
        all_chunks_used.extend(diagnosis_docs)
        
        # Format medical guide information without exposing sources
        diagnosis_content = "\n\n".join([doc['text'] for doc in diagnosis_docs])
        
        short_diagnosis_prompt = f'''Patient information:
Initial complaint: {initial_complaint}
Key findings: {"; ".join(key_findings)}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{diagnosis_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain sufficient information for a specific aspect, simply say the guide doesn't provide that information.
4. Format your response to be helpful to a community health worker exactly as shown in the medical guide.
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

What likely conditions might explain the patient's symptoms, based ONLY on the medical guide information?
'''

        diagnosis = get_openai_completion(
            prompt=short_diagnosis_prompt,
            max_tokens=300,
            temperature=0.2
        )
        
        # Get treatment recommendations in separate calls
        treatment_parts = []
        treatment_docs = []
        
        # 1. Immediate Care
        immediate_embedding = get_embedding_batch([initial_complaint + " immediate care steps"])[0]
        immediate_docs = vectorQuotesWithSource(immediate_embedding, index, top_k=2)
        treatment_docs.extend(immediate_docs)
        all_chunks_used.extend(immediate_docs)
        
        if immediate_docs:
            immediate_content = "\n\n".join([doc['text'] for doc in immediate_docs])
            immediate_prompt = f'''THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{immediate_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain sufficient information for immediate care, simply say the guide doesn't provide that information.
4. Present the immediate care steps exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, what immediate care steps should be taken?
'''
            
            immediate_care = get_openai_completion(
                prompt=immediate_prompt,
                max_tokens=200,
                temperature=0.2
            )
            treatment_parts.append("Immediate Care Steps:\n" + immediate_care)
        
        # 2. Home Care
        home_embedding = get_embedding_batch([initial_complaint + " home care advice"])[0]
        home_docs = vectorQuotesWithSource(home_embedding, index, top_k=2)
        treatment_docs.extend(home_docs)
        all_chunks_used.extend(home_docs)
        
        if home_docs:
            home_content = "\n\n".join([doc['text'] for doc in home_docs])
            home_prompt = f'''THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{home_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain specific home care information, simply say the guide doesn't provide that information.
4. Present the home care instructions exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, what home care instructions should be provided?
'''
            
            home_care = get_openai_completion(
                prompt=home_prompt,
                max_tokens=200,
                temperature=0.2
            )
            treatment_parts.append("\nHome Care:\n" + home_care)
        
        # 3. When to Refer 
        referral_embedding = get_embedding_batch([initial_complaint + " when to refer"])[0]
        referral_docs = vectorQuotesWithSource(referral_embedding, index, top_k=2)
        treatment_docs.extend(referral_docs)
        all_chunks_used.extend(referral_docs)
        
        referral_content = "\n\n".join([doc['text'] for doc in (referral_docs if referral_docs else diagnosis_docs)])
        referral_prompt = f'''THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{referral_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain specific referral criteria, simply say the guide doesn't provide that information.
4. Present the referral guidance exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and warning signs as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, when should this patient be referred to a higher level facility?
'''
        
        referral_guidance = get_openai_completion(
            prompt=referral_prompt,
            max_tokens=200,
            temperature=0.2
        )
        treatment_parts.append("\nReferral Guidance:\n" + referral_guidance)
        
        # Combine all parts
        treatment = "\n".join(treatment_parts)
        
        # Filter duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in diagnosis_docs + treatment_docs:
            if citation['id'] not in seen:
                seen.add(citation['id'])
                unique_citations.append(citation)
        
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": unique_citations,
            "chunks_used": all_chunks_used  # Include all chunks for reference
        }
            
    except Exception as e:
        print(f"Error in diagnosis/treatment: {e}")
        try:
            # Fallback to simpler request with explicit instructions not to use pretrained knowledge
            minimal_prompt = f'''IMPORTANT: You can ONLY use information from the WHO medical guide for community health workers for this response.

For a patient with '{initial_complaint}', what does the guide recommend?

If the medical guide doesn't contain relevant information about this condition, simply state that the guide doesn't provide specific information for this complaint and the patient should be referred to a health facility.

DO NOT use any medical knowledge from your pretraining.
'''
            
            fallback_diagnosis = get_openai_completion(
                prompt=minimal_prompt,
                max_tokens=150,
                temperature=0.2
            )
            
            return {
                "diagnosis": fallback_diagnosis,
                "treatment": "The medical guide doesn't provide specific treatment information for this condition. Please refer the patient to a health facility for proper evaluation and treatment.",
                "citations": [],
                "chunks_used": []
            }
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return {
                "diagnosis": "Error retrieving information from the medical guide. Please refer the patient to a health facility for proper evaluation.",
                "treatment": "Error retrieving information from the medical guide. Please refer the patient to a health facility for proper evaluation.",
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

""""""






def main():
    try:
        # Initialize/clear the global arrays
        global structured_questions_array, examination_history
        structured_questions_array = []
        examination_history = []
        
        initial_responses = []
        followup_responses = []
        exam_responses = []
        
        print("\nMedical Assessment Initial Questions")
        print("===================================")
        
        # Initial Questions Loop
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
                                initial_responses.append({
                                    "question": question['question'],
                                    "answer": [a.strip() for a in answers],
                                    "type": question['type']
                                })
                                break
                        else:
                            if validate_mc_input(answer, question['options']):
                                initial_responses.append({
                                    "question": question['question'],
                                    "answer": [answer],
                                    "type": question['type']
                                })
                                break
                    else:
                        if validate_mc_input(answer, question['options']):
                            if answer == "5":
                                custom_answer = input("Please specify: ").strip()
                                initial_responses.append({
                                    "question": question['question'],
                                    "answer": custom_answer,
                                    "type": question['type']
                                })
                            else:
                                selected_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                                initial_responses.append({
                                    "question": question['question'],
                                    "answer": selected_text,
                                    "type": question['type']
                                })
                            break
                
                elif question['type'] == 'NUM':
                    answer = input(f"Enter a number between {question['range']['min']} and {question['range']['max']}: ")
                    if validated_num := validate_num_input(answer, question['range']):
                        initial_responses.append({
                            "question": question['question'],
                            "answer": validated_num,
                            "type": question['type']
                        })
                        break
                        
                elif question['type'] == 'FREE':
                    answer = input("Enter your response (type your answer and press Enter): ").strip()
                    if answer:
                        initial_responses.append({
                            "question": question['question'],
                            "answer": answer,
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
                
                # Modified prompt to use #: format
                prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"}

Recommend ONE essential examination in this format (should not be first world exams like MRI, CT, Colonoscopy, etc.):
[Examination name]
[Procedure to perform the examination. Make sure this is detialed enough for a medical professional to understand and conduct]
#:[First possible finding]
#:[Second possible finding]
#:[Third possible finding]
#:[Fourth possible finding]'''

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
                        answer = input("Enter the finding (enter the number): ").strip()
                        
                        if validate_mc_input(answer, options):
                            # Store the examination data
                            store_examination(examination_text, int(answer))
                            
                            if answer == "5":
                                custom_result = input("Please specify the finding: ").strip()
                                exam_responses.append({
                                    "examination": examination,
                                    "result": custom_result,
                                    "type": "EXAM",
                                    "citations": exam_citations[-5:]
                                })
                            else:
                                selected_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                                exam_responses.append({
                                    "examination": examination,
                                    "result": selected_text,
                                    "type": "EXAM",
                                    "citations": exam_citations[-5:]
                                })
                            
                            # Print global arrays after each answer
                            print("\nCurrent Structured Questions Array:")
                            print("================================")
                            print(json.dumps(structured_questions_array, indent=2))
                            
                            print("\nCurrent Examination History:")
                            print("=========================")
                            print(json.dumps(examination_history, indent=2))
                            break
                            
                        print("Invalid input, please try again.")
                        
                except ValueError as e:
                    print(f"Error parsing examination: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error generating examination: {e}")
                continue

        # Get and print diagnosis and treatment
        results = get_diagnosis_and_treatment(
            initial_responses,
            followup_responses,
            exam_responses
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
        
        # Print final global arrays
        print("\nFinal Structured Questions Array:")
        print("==============================")
        print(json.dumps(structured_questions_array, indent=2))
        
        extract_question_and_options(structured_questions_array, 1)
        
        print("\nFinal Examination History:")
        print("========================")
        print(json.dumps(examination_history, indent=2))

        extract_examination_and_options(examination_history, 1)
        
    except KeyboardInterrupt:
        print("\nAssessment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()