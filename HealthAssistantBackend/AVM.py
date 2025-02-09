from typing import Dict, List, Any, Optional, Tuple
import os
import json
from dotenv import load_dotenv
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

# Initial screening questions (to be provided externally if needed)
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
        # Return an empty list for each text on error.
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
    for chunk, _ in zip(text_chunks, embeddings):
        if chunk not in seen_content:
            compressed_chunks.append(chunk)
            seen_content.add(chunk)
    
    return "\n".join(compressed_chunks[:5]), []

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

# --- GLOBAL ARRAYS FOR STORING STRUCTURED DATA ---
structured_questions_array: List[Dict[str, Any]] = []
examination_history: List[Dict[str, Any]] = []

def store_question(question_data: dict):
    """Store a question in the global array."""
    global structured_questions_array
    structured_questions_array.append(question_data)

def parse_question_data(question: str, options: List[Dict[str, Any]], answer: str, matrix_output: dict, citations: List[Dict[str, Any]]) -> dict:
    """Parse a single question's data into a structured format."""
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    selected_idx = next((i+1 for i, opt in enumerate(options) if opt['text'] == answer), None)
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" for cite in citations] if citations else []
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

def process_with_matrix(current_text: str, previous_responses: List[Dict[str, Any]], 
                        context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced error handling."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        state = matrix.state_encoder.encode_state([], previous_responses, current_text)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.squeeze(1)
        # Although optimist_view and pessimist_view are computed here, we simply call process_state.
        matrix_output = matrix.process_state([], previous_responses, current_text)
        return matrix_output
    except Exception as e:
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5}
        }

def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> bool:
    """Judge if the current question is too similar using MATRIX."""
    if len(followup_responses) >= MATRIXConfig.MAX_QUESTIONS:
        return True
    if not followup_responses:
        return False
    try:
        matrix_output = process_with_matrix(current_question, followup_responses)
        should_stop = (
            matrix_output["confidence"] > MATRIXConfig.SIMILARITY_THRESHOLD or
            len(followup_responses) >= MATRIXConfig.MAX_FOLLOWUPS or
            matrix_output["weights"]["optimist"] > 0.7
        )
        return should_stop
    except Exception as e:
        return len(followup_responses) >= 5

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> bool:
    """Judge examination similarity with improved duplicate detection."""
    if not previous_exams:
        return False
    try:
        if len(previous_exams) >= MATRIXConfig.MAX_EXAMS:
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
                return True
            if prev_procedure and current_procedure:
                words1 = set(prev_procedure.split())
                words2 = set(current_procedure.split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                if similarity > 0.7:
                    return True
        matrix_output = process_with_matrix(current_exam, previous_exams)
        should_end = (
            matrix_output["confidence"] > MATRIXConfig.EXAM_SIMILARITY_THRESHOLD or
            len(previous_exams) >= MATRIXConfig.MAX_EXAMS or
            matrix_output["weights"]["optimist"] > 0.8
        )
        return should_end
    except Exception as e:
        return len(previous_exams) >= MATRIXConfig.MAX_EXAMS

def parse_examination_text(examination_text: str) -> Tuple[str, List[str]]:
    """
    Parse examination text using "#:" delimiter.
    Returns a tuple of (examination text, list of options).
    """
    parts = examination_text.split("#:")
    if len(parts) < 2:
        raise ValueError("Invalid examination format - missing '#:' delimiter")
    examination = parts[0].strip()
    options = [opt.strip() for opt in parts[1:] if opt.strip()]
    return examination, options

def store_examination(examination_text: str, selected_option: int):
    """Store examination data in the global examination history."""
    global examination_history
    try:
        examination, options = parse_examination_text(examination_text)
        examination_entry = {
            "examination": examination,
            "options": options,
            "selected_option": selected_option
        }
        examination_history.append(examination_entry)
    except Exception as e:
        pass  # In a pure data–processing function, you might log the error elsewhere

# === PURE DATA–DRIVEN FUNCTIONS (NO INTERACTIVE I/O) ===

def run_followup_assessment(initial_responses: List[Dict[str, Any]], followup_answers: List[str]) -> List[Dict[str, Any]]:
    """
    Run the follow-up assessment using a provided list of follow-up answer strings.
    For each provided answer, a new follow-up question is generated (using GPT via groq_client)
    and the answer is recorded. The process stops when judge() returns True or when the answers are exhausted.
    Returns the complete list of follow-up responses.
    """
    followup_responses: List[Dict[str, Any]] = []
    index = pc.Index("final-asha")
    # Extract the initial complaint from the initial responses
    initial_complaint = next((resp['answer'] for resp in initial_responses
                              if resp['question'] == "Please describe what brings you here today"), "")
    for answer_text in followup_answers:
        # Build context from initial complaint and previous follow-up responses
        context = f"Initial complaint: {initial_complaint}\n"
        if followup_responses:
            context += "Previous responses:\n"
            for resp in followup_responses:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        if not relevant_docs:
            # If no relevant docs are found, skip this iteration.
            continue
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses]) if followup_responses else "No previous questions yet"
        prompt = (
            f'Based on the patient\'s initial complaint: "{initial_complaint}"\n\n'
            f'Previous questions asked:\n{previous_questions}\n\n'
            f'Relevant medical context:\n{combined_context}\n\n'
            'Generate ONE focused, relevant follow-up question that is different from the previous questions. '
            'Follow standard medical assessment order:\n'
            '1. Duration and onset\n'
            '2. Characteristics and severity\n'
            '3. Associated symptoms\n'
            '4. Impact on daily life\n\n'
            'Return only the question text.'
        )
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        question = completion.choices[0].message.content.strip()
        # Generate options for the question
        options_prompt = (
            f'Generate 4 concise answers for: "{question}"\n'
            'Clear, mutually exclusive options.\n'
            'Return each option on a new line.'
        )
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
                if len(text) >= 2 and text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                options.append({"id": i+1, "text": text})
        # Append a standard "Other" option.
        options.append({"id": 5, "text": "Other (please specify)"})
        matrix_output = process_with_matrix(question, followup_responses)
        # Record the provided answer (here, we assume the answer text is valid)
        followup_responses.append({
            "question": question,
            "answer": answer_text,
            "type": "MC",
            "citations": relevant_docs[-5:]
        })
        parse_question_data(question, options, answer_text, matrix_output, relevant_docs[-5:])
        if judge(followup_responses, question):
            break
    return followup_responses

def run_exam_assessment(initial_responses: List[Dict[str, Any]], 
                        followup_responses: List[Dict[str, Any]],
                        exam_answers: List[str]) -> List[Dict[str, Any]]:
    """
    Run the examination phase using a provided list of exam answer strings.
    Returns the list of exam responses.
    """
    exam_responses: List[Dict[str, Any]] = []
    index = pc.Index("final-asha")
    initial_complaint = next((resp['answer'] for resp in initial_responses
                              if resp['question'] == "Please describe what brings you here today"), "")
    # Track key symptoms based on follow-up answers
    symptoms = set()
    for resp in followup_responses:
        ans = str(resp['answer']).lower()
        for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                        'headache', 'nausea', 'dizziness', 'rash']:
            if symptom in ans:
                symptoms.add(symptom)
    for answer_text in exam_answers:
        context = (
            f"Initial complaint: {initial_complaint}\n"
            f"Key symptoms: {', '.join(symptoms)}\n"
            f"Previous findings: {str([exam['examination'] for exam in exam_responses]) if exam_responses else 'None'}"
        )
        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index)
        if not relevant_docs:
            continue
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        prompt = (
            f'Based on:\nInitial complaint: "{initial_complaint}"\n'
            f'Key symptoms: {", ".join(symptoms)}\n\n'
            f'Previous exams: {str([exam["examination"] for exam in exam_responses]) if exam_responses else "None"}\n\n'
            'Recommend ONE essential examination in this format (avoid high–tech exams such as MRI, CT, etc.):\n'
            '[Examination name and procedure]\n'
            '#:[First possible finding]\n'
            '#:[Second possible finding]\n'
            '#:[Third possible finding]\n'
            '#:[Fourth possible finding]'
        )
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200
        )
        examination_text = completion.choices[0].message.content.strip()
        if judge_exam(exam_responses, examination_text):
            break
        try:
            examination, option_texts = parse_examination_text(examination_text)
        except ValueError:
            continue
        options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
        options.append({"id": 5, "text": "Other (please specify)"})
        # Use the provided answer for the exam phase.
        exam_responses.append({
            "examination": examination,
            "result": answer_text,
            "type": "EXAM",
            "citations": relevant_docs[-5:]
        })
        # Optionally, store the examination (the selected option is inferred from the answer).
        try:
            selected_option = next(i+1 for i, opt in enumerate(options) if opt["text"] == answer_text)
        except StopIteration:
            selected_option = 5
        store_examination(examination_text, selected_option)
    return exam_responses

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                                followup_responses: List[Dict[str, Any]], 
                                exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get diagnosis and treatment with citations using a series of GPT calls."""
    try:
        initial_complaint = next((resp['answer'] for resp in initial_responses
                                  if resp['question'] == "Please describe what brings you here today"), "")
        key_findings = []
        for resp in followup_responses + exam_responses:
            if isinstance(resp.get('answer'), str):
                key_findings.append(resp['answer'])
        key_findings = key_findings[-3:]
        index = pc.Index("final-asha")
        diagnosis_embedding = get_embedding_batch([initial_complaint + " diagnosis"])[0]
        diagnosis_docs = vectorQuotesWithSource(diagnosis_embedding, index, top_k=2)
        diagnosis_context = " ".join([doc["text"] for doc in diagnosis_docs])
        short_diagnosis_prompt = (
            f"Patient complaint: {initial_complaint}\n"
            f"Key findings: {'; '.join(key_findings)}\n"
            f"Reference: {diagnosis_context[:200]}\n\n"
            "List top 3-4 possible diagnoses based on symptoms."
        )
        diagnosis_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": short_diagnosis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=100
        )
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        # Treatment recommendations in separate calls
        treatment_parts = []
        treatment_docs = []
        # Immediate Care
        immediate_embedding = get_embedding_batch([initial_complaint + " immediate care steps"])[0]
        immediate_docs = vectorQuotesWithSource(immediate_embedding, index, top_k=1)
        treatment_docs.extend(immediate_docs)
        if immediate_docs:
            immediate_prompt = (
                f"Based on: {immediate_docs[0]['text'][:200]}\n"
                f"Provide 2-3 immediate care steps for {initial_complaint}."
            )
            immediate_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": immediate_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("Immediate Care:\n" + immediate_completion.choices[0].message.content.strip())
        # Medications
        med_embedding = get_embedding_batch([initial_complaint + " medications treatment"])[0]
        med_docs = vectorQuotesWithSource(med_embedding, index, top_k=1)
        treatment_docs.extend(med_docs)
        if med_docs:
            med_prompt = (
                f"Based on: {med_docs[0]['text'][:200]}\n"
                f"List 2-3 key medications or supplements for {initial_complaint}."
            )
            med_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": med_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("\nMedications/Supplements:\n" + med_completion.choices[0].message.content.strip())
        # Home Care
        home_embedding = get_embedding_batch([initial_complaint + " home care follow up"])[0]
        home_docs = vectorQuotesWithSource(home_embedding, index, top_k=1)
        treatment_docs.extend(home_docs)
        if home_docs:
            home_prompt = (
                f"Based on: {home_docs[0]['text'][:200]}\n"
                f"List 2-3 home care instructions for {initial_complaint}."
            )
            home_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": home_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=100
            )
            treatment_parts.append("\nHome Care:\n" + home_completion.choices[0].message.content.strip())
        treatment = "\n".join(treatment_parts)
        citations = diagnosis_docs + treatment_docs
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": citations
        }
    except Exception as e:
        # In case of error, fall back to a simpler prompt
        try:
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
            return {
                "diagnosis": "Error generating diagnosis",
                "treatment": "Error generating treatment",
                "citations": []
            }

def run_assessment(initial_responses: List[Dict[str, Any]],
                   followup_answers: List[str],
                   exam_answers: List[str]) -> Dict[str, Any]:
    """
    High-level function that runs the complete assessment process.
    It expects:
      - initial_responses: A list of dictionaries with the initial screening answers.
      - followup_answers: A list of answer strings for the follow-up phase.
      - exam_answers: A list of answer strings for the examination phase.
    Returns a dictionary with:
      - The original responses.
      - Follow-up responses.
      - Exam responses.
      - Diagnosis and treatment.
      - The structured questions array.
      - The examination history.
    """
    # Reset global arrays
    global structured_questions_array, examination_history
    structured_questions_array = []
    examination_history = []

    followup_responses = run_followup_assessment(initial_responses, followup_answers)
    exam_responses = run_exam_assessment(initial_responses, followup_responses, exam_answers)
    diagnosis_treatment = get_diagnosis_and_treatment(initial_responses, followup_responses, exam_responses)
    
    return {
        "initial_responses": initial_responses,
        "followup_responses": followup_responses,
        "exam_responses": exam_responses,
        "diagnosis": diagnosis_treatment["diagnosis"],
        "treatment": diagnosis_treatment["treatment"],
        "citations": diagnosis_treatment["citations"],
        "structured_questions_array": structured_questions_array,
        "examination_history": examination_history
    }
