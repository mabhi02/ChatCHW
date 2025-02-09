from typing import Dict, List, Any, Optional, Tuple
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

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's API."""
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
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
    
    compressed_chunks = []
    seen_content = set()
    
    for chunk, embedding in zip(text_chunks, embeddings):
        if chunk not in seen_content:
            compressed_chunks.append(chunk)
            seen_content.add(chunk)
    
    return "\n".join(compressed_chunks[:5]), []

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                       context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        state = matrix.state_encoder.encode_state([], previous_responses, current_text)
        
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.squeeze(1)
            
        matrix_output = matrix.process_state([], previous_responses, current_text)
        return matrix_output
        
    except Exception as e:
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5}
        }


def generate_question_with_options(input_data: List[Dict[str, Any]], previous_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a follow-up question based on context."""
    try:
        initial_complaint = next((resp['answer'] for resp in input_data 
                            if resp['question'] == "Please describe what brings you here today"), "")

        # Build context from previous responses
        context = "Previous responses:\n"
        for resp in previous_responses:
            context += f"Q: {resp['question']}\nA: {resp['answer']}\n"

        prompt = f'''Based on initial complaint: "{initial_complaint}"
        Previous conversation: {context}
        
        Generate ONE focused follow-up question that helps understand the patient's condition better.
        The question should not be too similar to previous ones.
        Follow medical assessment order:
        1. Duration and onset
        2. Characteristics and severity
        3. Associated symptoms
        4. Impact on daily life
        
        Return only the question text.'''
        
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        
        question = completion.choices[0].message.content.strip()
        
        # Generate contextual options
        options_prompt = f'''For the medical question: "{question}"
        Generate 4 distinct answer options that:
        - Are mutually exclusive
        - Cover the likely range of responses
        - Are clear and concise
        
        Format each option on a new line.'''
        
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
                if text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                options.append({"id": i+1, "text": text})
        
        options.append({"id": 5, "text": "Other (please specify)"})
        
        return {
            "question": question,
            "type": "MC",
            "options": options
        }

    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def get_followup_questions(initial_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate and process follow-up questions."""
    try:
        index = pc.Index("final-asha")
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Don't move to examination phase until we've gathered enough context
        prompt = f'''Based on the initial complaint: "{initial_complaint}"
        
        Generate a focused follow-up question to better understand the patient's condition.
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
            max_tokens=150
        )
        
        question = completion.choices[0].message.content.strip()
        
        # Generate options dynamically based on the question
        options_prompt = f'''Generate 4 appropriate answers for: "{question}"
        Each answer should be clear and distinct.
        Return each option on a new line.'''
        
        options_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": options_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=100
        )
        
        options = []
        for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
            if opt.strip():
                options.append({"id": i+1, "text": opt.strip()})
        
        options.append({"id": 5, "text": "Other (please specify)"})
        
        return {
            "question": question,
            "type": "MC",
            "options": options
        }
            
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return {
            "question": "Unable to generate follow-up questions",
            "type": "MC",
            "options": [],
            "matrix_data": [],
            "citations": []
        }
        
def get_followup_exams(initial_responses: List[Dict[str, Any]], 
                      followup_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate and process examination recommendations."""
    try:
        index = pc.Index("final-asha")
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Track symptoms
        symptoms = set()
        for resp in followup_responses:
            answer = resp['answer'].lower()
            for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling']:
                if symptom in answer:
                    symptoms.add(symptom)
        
        context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}"""

        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            return []
        
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        # Generate examination recommendations
        prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}
Context: {combined_context}

Recommend 3 essential examinations as a JSON array. Each with:
1. Name
2. Procedure
3. Expected findings'''
        
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300
        )
        
        examinations = []
        try:
            results = json.loads(completion.choices[0].message.content)
            for exam in results:
                examinations.append({
                    "examination": exam['name'],
                    "procedure": exam['procedure'],
                    "findings": exam['findings'],
                    "citations": relevant_docs[-5:]
                })
        except:
            # Fallback if JSON parsing fails
            examinations = [{
                "examination": "Basic Physical Examination",
                "procedure": "Standard physical assessment",
                "findings": ["Normal", "Abnormal", "Requires further testing"],
                "citations": relevant_docs[-5:]
            }]
        
        return examinations
            
    except Exception as e:
        return []

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              followup_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate final diagnosis and treatment recommendations."""
    try:
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Compile key findings
        findings = []
        for resp in followup_responses + exam_responses:
            if isinstance(resp.get('answer'), str):
                findings.append(resp['answer'])
        findings = findings[-3:]
        
        index = pc.Index("final-asha")
        embedding = get_embedding_batch([initial_complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            return {
                "diagnosis": "Unable to generate diagnosis",
                "treatment": "Please consult a healthcare provider",
                "citations": []
            }
            
        context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        # Generate complete assessment
        prompt = f'''Patient complaint: {initial_complaint}
Key findings: {"; ".join(findings)}
Context: {context}

Provide a complete assessment in JSON format with:
1. Top 3 possible diagnoses
2. Immediate care steps
3. Medications/supplements
4. Home care instructions'''
        
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=500
        )
        
        try:
            results = json.loads(completion.choices[0].message.content)
            
            return {
                "diagnosis": results['diagnoses'],
                "treatment": {
                    "immediate_care": results['immediate_care'],
                    "medications": results['medications'],
                    "home_care": results['home_care']
                },
                "citations": relevant_docs,
                "matrix_output": process_with_matrix(
                    str(results['diagnoses']),
                    initial_responses + followup_responses
                )
            }
        except:
            return {
                "diagnosis": "Error processing diagnosis",
                "treatment": "Please consult a healthcare provider",
                "citations": relevant_docs,
                "matrix_output": {
                    "confidence": 0.5,
                    "selected_agent": "optimist",
                    "weights": {"optimist": 0.5, "pessimist": 0.5}
                }
            }
            
    except Exception as e:
        return {
            "diagnosis": "System error in diagnosis generation",
            "treatment": "Please consult a healthcare provider",
            "citations": [],
            "matrix_output": {
                "confidence": 0.5,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.5, "pessimist": 0.5}
            }
        }

def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> bool:
    """Judge if the current question is too similar using MATRIX."""
    if len(followup_responses) < 3:
        return False
        
    try:
        matrix_output = process_with_matrix(
            current_question, 
            followup_responses
        )
        
        should_stop = (
            (len(followup_responses) >= 5 and matrix_output["confidence"] > 0.7) or
            len(followup_responses) >= MATRIXConfig.MAX_FOLLOWUPS or
            (matrix_output["weights"]["optimist"] > 0.8 and len(followup_responses) >= 4)
        )
        
        return should_stop
        
    except Exception as e:
        return len(followup_responses) >= 7

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> Dict[str, Any]:
    """Judge examination similarity with improved duplicate detection."""
    if not previous_exams:
        return {
            "should_stop": False,
            "reason": "First examination",
            "matrix_output": None
        }
        
    try:
        if len(previous_exams) >= MATRIXConfig.MAX_EXAMS:
            return {
                "should_stop": True,
                "reason": "Maximum examinations reached",
                "matrix_output": None
            }
            
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
                return {
                    "should_stop": True,
                    "reason": f"Similar examination already performed: {current_exam_name}",
                    "matrix_output": None
                }
                
            if len(prev_procedure) > 0 and len(current_procedure) > 0:
                words1 = set(prev_procedure.split())
                words2 = set(current_procedure.split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if similarity > 0.7:
                    return {
                        "should_stop": True,
                        "reason": "Very similar procedure already performed",
                        "matrix_output": None
                    }
        
        matrix_output = process_with_matrix(
            current_exam, 
            previous_exams
        )
        
        should_end = (
            matrix_output["confidence"] > MATRIXConfig.EXAM_SIMILARITY_THRESHOLD or
            len(previous_exams) >= MATRIXConfig.MAX_EXAMS or
            matrix_output["weights"]["optimist"] > 0.8
        )
        
        return {
            "should_stop": should_end,
            "reason": "MATRIX analysis complete",
            "matrix_output": matrix_output
        }
                
    except Exception as e:
        return {
            "should_stop": len(previous_exams) >= MATRIXConfig.MAX_EXAMS,
            "reason": "Fallback to basic method",
            "matrix_output": None
        }