from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from groq import Groq
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

# Load environment variables and initialize clients
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

def compress_medical_context(responses: List[Dict[str, Any]], 
                           embeddings: Optional[List[List[float]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Compress medical context by using embeddings to find key information."""
    text_chunks = []
    for resp in responses:
        if isinstance(resp.get('answer'), str):
            text_chunks.append(f"{resp['question']}: {resp['answer']}")
        elif isinstance(resp.get('answer'), list):
            text_chunks.append(f"{resp['question']}: {', '.join(resp['answer'])}")
    
    if not embeddings:
        embeddings = get_embedding_batch(text_chunks)
    
    # Use embeddings to find most relevant chunks
    compressed_chunks = []
    seen_content = set()
    
    for chunk, embedding in zip(text_chunks, embeddings):
        if chunk not in seen_content:
            compressed_chunks.append(chunk)
            seen_content.add(chunk)
    
    return "\n".join(compressed_chunks[:5]), []

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

def get_initial_responses() -> List[Dict[str, Any]]:
    """Gather responses for initial screening questions."""
    responses = []
    
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
                            responses.append({
                                "question": question['question'],
                                "answer": [a.strip() for a in answers],
                                "type": question['type']
                            })
                            break
                    else:
                        if validate_mc_input(answer, question['options']):
                            responses.append({
                                "question": question['question'],
                                "answer": [answer],
                                "type": question['type']
                            })
                            break
                else:
                    if validate_mc_input(answer, question['options']):
                        if answer == "5":
                            custom_answer = input("Please specify: ").strip()
                            responses.append({
                                "question": question['question'],
                                "answer": custom_answer,
                                "type": question['type']
                            })
                        else:
                            selected_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                            responses.append({
                                "question": question['question'],
                                "answer": selected_text,
                                "type": question['type']
                            })
                        break
                
            elif question['type'] == 'NUM':
                answer = input(f"Enter a number between {question['range']['min']} and {question['range']['max']}: ")
                if validated_num := validate_num_input(answer, question['range']):
                    responses.append({
                        "question": question['question'],
                        "answer": validated_num,
                        "type": question['type']
                    })
                    break
                    
            elif question['type'] == 'FREE':
                answer = input("Enter your response (type your answer and press Enter): ").strip()
                if answer:
                    responses.append({
                        "question": question['question'],
                        "answer": answer,
                        "type": question['type']
                    })
                    break
            
            print("Invalid input, please try again.")
    
    return responses


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
        
        print(f"\nQuestion Assessment:")
        print(f"- Confidence: {matrix_output['confidence']:.2f}")
        print(f"- Selected Agent: {matrix_output['selected_agent']}")
        print(f"- Optimist Weight: {matrix_output['weights']['optimist']:.2f}")
        print(f"- Pessimist Weight: {matrix_output['weights']['pessimist']:.2f}")
        
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

def get_followup_questions(initial_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate and ask follow-up questions with enhanced processing and citations."""
    followup_responses = []
    initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
    
    print("\nBased on your responses, I'll ask some follow-up questions.")
    
    index = pc.Index("final-asha")
    question_citations = []
    
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
                
            # Track citations
            question_citations.extend(relevant_docs)
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
            
            previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
            prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
            
            Previous questions asked:
            {previous_questions if followup_responses else "No previous questions yet"}
            
            Relevant medical context:
            {combined_context}
            
            Generate ONE focused, relevant follow-up question that is different from the previous questions.
            Follow standard medical assessment order:
            1. Duration and onset
            2. Characteristics and severity
            3. Associated symptoms
            4. Impact on daily life
            
            Return only the question text.'''
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=150  # Limit response size
            )
            
            question = completion.choices[0].message.content.strip()
            
            # Use MATRIX to judge similarity
            if judge(followup_responses, question):
                print("\nSufficient information gathered. Moving to next phase...")
                break
            
            # Generate shorter options prompt
            options_prompt = f'''Generate 4 concise answers for: "{question}"
            Clear, mutually exclusive options.
            Return each option on a new line (1-4).'''
            
            options_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": options_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100  # Limit response size
            )
            
            options = []
            for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
                if opt.strip():
                    text = opt.strip()
                    if text[0].isdigit() and text[1] in ['.','-',')']:
                        text = text[2:].strip()
                    options.append({"id": i+1, "text": text})
            
            options.append({"id": 5, "text": "Other (please specify)"})
            
            print(f"\n{question}")
            print_options(options)
            
            while True:
                answer = input("Enter your choice (enter the number): ").strip()
                
                if validate_mc_input(answer, options):
                    if answer == "5":
                        custom_answer = input("Please specify your answer: ").strip()
                        followup_responses.append({
                            "question": question,
                            "answer": custom_answer,
                            "type": "MC",
                            "citations": question_citations[-5:]  # Keep last 5 citations
                        })
                    else:
                        selected_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                        followup_responses.append({
                            "question": question,
                            "answer": selected_text,
                            "type": "MC",
                            "citations": question_citations[-5:]
                        })
                    break
                print("Invalid input, please try again.")
                
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            continue
            
    return followup_responses

def get_followup_exams(initial_responses: List[Dict[str, Any]], 
                      followup_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate and recommend examinations with citations."""
    exam_responses = []
    initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
    
    print("\nBased on your responses, I'll recommend appropriate examinations.")
    
    index = pc.Index("final-asha")
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
            
            # Track citations
            exam_citations.extend(relevant_docs)
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
            
            # Shorter examination prompt
            prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"}

Recommend ONE essential examination:
1. Addresses reported symptoms
2. Uses basic equipment
3. Not yet performed

Format:
Examination: [name]
Procedure: [steps]'''

            completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200
            )
            
            examination = completion.choices[0].message.content.strip()
            
            if judge_exam(exam_responses, examination):
                break
                
            # Shorter findings prompt
            results_prompt = f'''For examination:
"{examination}"

Generate 4 possible findings.
Include normal and abnormal results.
One per line (1-4).'''
            
            results_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": results_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            
            options = []
            for i, opt in enumerate(results_completion.choices[0].message.content.strip().split('\n')):
                if opt.strip():
                    text = opt.strip()
                    if text[0].isdigit() and text[1] in ['.','-',')']:
                        text = text[2:].strip()
                    options.append({"id": i+1, "text": text})
            
            options.append({"id": 5, "text": "Other (please specify)"})
            
            print(f"\nRecommended Examination:")
            print(examination)
            print("\nSelect the finding:")
            print_options(options)
            
            while True:
                answer = input("Enter the finding (enter the number): ").strip()
                
                if validate_mc_input(answer, options):
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
                    break
                print("Invalid input, please try again.")
                
        except Exception as e:
            print(f"Error generating examination: {e}")
            continue
            
    return exam_responses

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              followup_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get diagnosis and treatment with citations, using smaller chunked requests."""
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

        diagnosis_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": short_diagnosis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=100
        )
        
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        
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
            
            immediate_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": immediate_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("Immediate Care:\n" + immediate_completion.choices[0].message.content.strip())
        
        # 2. Medications
        med_embedding = get_embedding_batch([initial_complaint + " medications treatment"])[0]
        med_docs = vectorQuotesWithSource(med_embedding, index, top_k=1)
        treatment_docs.extend(med_docs)
        
        if med_docs:
            med_prompt = f'''Based on: {med_docs[0]["text"][:200]}
List 2-3 key medications or supplements for {initial_complaint}.'''
            
            med_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": med_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("\nMedications/Supplements:\n" + med_completion.choices[0].message.content.strip())
        
        # 3. Home Care
        home_embedding = get_embedding_batch([initial_complaint + " home care follow up"])[0]
        home_docs = vectorQuotesWithSource(home_embedding, index, top_k=1)
        treatment_docs.extend(home_docs)
        
        if home_docs:
            home_prompt = f'''Based on: {home_docs[0]["text"][:200]}
List 2-3 home care instructions for {initial_complaint}.'''
            
            home_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": home_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("\nHome Care:\n" + home_completion.choices[0].message.content.strip())
        
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
            fallback_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": minimal_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=50
            )
            return {
                "diagnosis": fallback_completion.choices[0].message.content.strip(),
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
            
    except Exception as e:
        print(f"Error in diagnosis/treatment: {e}")
        return {
            "diagnosis": "Error generating diagnosis",
            "treatment": "Error generating treatment",
            "citations": []
        }



def main():
    try:
        initial_responses = get_initial_responses()
        print("\nThank you for providing your information. Here's what we recorded:\n")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")
        
        followup_responses = get_followup_questions(initial_responses)
        print("\nFollow-up responses recorded:\n")
        for resp in followup_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")
            if "citations" in resp:
                print("Sources consulted:")
                for cite in resp["citations"]:
                    print(f"- {cite['source']} (relevance: {cite['score']:.2f})")
            print()
        
        exam_responses = get_followup_exams(initial_responses, followup_responses)
        print("\nExamination findings recorded:\n")
        for exam in exam_responses:
            print(f"Examination: {exam['examination']}")
            print(f"Finding: {exam['result']}")
            if "citations" in exam:
                print("Sources consulted:")
                for cite in exam["citations"]:
                    print(f"- {cite['source']} (relevance: {cite['score']:.2f})")
            print()

        results = get_diagnosis_and_treatment(
            initial_responses,
            followup_responses,
            exam_responses
        )
        
        print("\nDiagnosis:")
        print("==========")
        print(results["diagnosis"])

        print("\nRecommended Treatment Plan:")
        print("=========================")
        print(results["treatment"])
        
        print("\nKey References:")
        print("==============")
        seen_sources = set()
        for citation in results["citations"]:
            if citation['source'] not in seen_sources:
                print(f"- {citation['source']} (relevance: {citation['score']:.2f})")
                seen_sources.add(citation['source'])
            
        return {
            "initial_responses": initial_responses,
            "followup_responses": followup_responses,
            "examination_responses": exam_responses,
            "diagnosis": results["diagnosis"],
            "treatment": results["treatment"],
            "citations": results["citations"]
        }
            
    except KeyboardInterrupt:
        print("\nAssessment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()