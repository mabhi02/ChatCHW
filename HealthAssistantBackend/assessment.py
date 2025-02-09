"""
assessment.py

This module exposes a collection of functions and a state‐holding class “AssessmentSession”
that you can use to drive a medical assessment flow from a frontend. It no longer uses interactive 
command‑line I/O. Instead, each method returns data (e.g. questions, options, diagnosis) that 
your frontend can render.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import json
import torch
import torch.nn as nn
import openai
from dotenv import load_dotenv

# Third‐party clients (assumed installed)
from groq import Groq
from pinecone import Pinecone

# Import custom MATRIX components (assumed available in your environment)
from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

# =============================================================================
# Environment and client initialization
# =============================================================================

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize MATRIX system and components
matrix = MATRIX()
decoder_tuner = DecoderTuner(matrix.meta_learner.decoder)
visualizer = AttentionVisualizer()
pattern_analyzer = PatternAnalyzer()

# =============================================================================
# Predefined initial screening questions (non-interactive)
# =============================================================================

questions_init: List[Dict[str, Any]] = [
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

# =============================================================================
# Helper functions
# =============================================================================

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using OpenAI's API.
    """
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        # In production, consider logging errors instead of printing.
        print(f"Error getting embeddings: {e}")
        return [[] for _ in texts]


def vectorQuotesWithSource(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search the vector database and return relevant matches with source information.
    """
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
    """
    Compress medical context by using embeddings to find key information.
    """
    text_chunks = []
    for resp in responses:
        answer = resp.get('answer')
        if isinstance(answer, str):
            text_chunks.append(f"{resp['question']}: {answer}")
        elif isinstance(answer, list):
            text_chunks.append(f"{resp['question']}: {', '.join(answer)}")
    
    if not embeddings:
        embeddings = get_embedding_batch(text_chunks)
    
    # Select the first few unique chunks
    seen = set()
    compressed_chunks = []
    for chunk in text_chunks:
        if chunk not in seen:
            compressed_chunks.append(chunk)
            seen.add(chunk)
    return "\n".join(compressed_chunks[:5]), []


def validate_num_input(value: str, range_data: Dict[str, int]) -> Optional[int]:
    """
    Validate numeric input against the given range.
    """
    try:
        num = int(value)
        if range_data['min'] <= num <= range_data['max']:
            return num
        return None
    except ValueError:
        return None


def validate_mc_input(value: str, options: List[Dict[str, Any]]) -> Optional[str]:
    """
    Validate multiple choice input against given options.
    """
    valid_ids = [str(opt['id']) for opt in options]
    return value if value in valid_ids else None


def parse_examination_text(examination_text: str) -> Tuple[str, List[str]]:
    """
    Parse an examination text using the "#:" delimiter.
    Returns a tuple of (examination text, list of options).
    """
    parts = examination_text.split("#:")
    if len(parts) < 2:
        raise ValueError("Invalid examination format - missing '#:' delimiter")
    examination = parts[0].strip()
    options = [opt.strip() for opt in parts[1:] if opt.strip()]
    return examination, options


def parse_question_data(question: str, options: List[Dict[str, Any]], answer: str,
                        matrix_output: Dict[str, Any], citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse a question's data into a structured format.
    Returns a dictionary containing the question, options, the selected option,
    pattern weights, confidence, and citations.
    """
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    selected_idx = next((i+1 for i, opt in enumerate(options) if opt['text'] == answer), None)
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" for cite in citations] if citations else []
    
    return {
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


def generate_question_with_options(input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a follow-up question with options based on patient input.
    Uses predefined templates to avoid API rate limits.
    """
    try:
        # Retrieve the initial complaint (if available)
        initial_complaint = next(
            (resp['answer'] for resp in input_data 
             if resp['question'] == "Please describe what brings you here today"),
            ""
        ).lower()

        # Define two sets of questions (for “pain” complaints or general complaints)
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

        questions = pain_questions if "pain" in initial_complaint else general_questions
        asked_questions = {resp.get('question', '') for resp in input_data if resp.get('question')}

        for question_data in questions:
            if question_data['question'] not in asked_questions:
                return {
                    "question": question_data['question'],
                    "options": question_data['options'],
                    "type": "MC"
                }

        # Fallback question if all have been used
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
    """
    Process input through the MATRIX system with enhanced error handling.
    """
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        # Encode the current state based on responses and current text.
        state = matrix.state_encoder.encode_state([], previous_responses, current_text)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.squeeze(1)
        optimist_view = matrix.optimist.evaluate(state, context_text if context_text else current_text)
        pessimist_view = matrix.pessimist.evaluate(state, context_text if context_text else current_text)
        matrix_output = matrix.process_state([], previous_responses, current_text)
        return matrix_output
    except Exception as e:
        print(f"Warning: MATRIX processing error: {e}")
        return {
            "confidence": 0.5,
            "selected_agent": "optimist",
            "weights": {"optimist": 0.5, "pessimist": 0.5}
        }


def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> bool:
    """
    Judge if the current question is too similar using the MATRIX system.
    """
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
        print(f"Warning: Similarity check fallback: {e}")
        return len(followup_responses) >= 5


def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> bool:
    """
    Judge examination similarity with improved duplicate detection.
    """
    if not previous_exams:
        return False
    try:
        if len(previous_exams) >= MATRIXConfig.MAX_EXAMS:
            return True

        # Extract key parts from the exam text
        exam_lines = current_exam.split('\n')
        current_exam_name = ""
        current_procedure = ""
        for line in exam_lines:
            if line.startswith("Examination:"):
                current_exam_name = line.split('Examination:')[1].strip().lower()
            elif line.startswith("Procedure:"):
                current_procedure = line.split('Procedure:')[1].strip().lower()

        for exam in previous_exams:
            prev_exam_lines = exam.get('examination', "").split('\n')
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
        print(f"Warning: Exam similarity fallback: {e}")
        return len(previous_exams) >= MATRIXConfig.MAX_EXAMS


# =============================================================================
# The AssessmentSession class
# =============================================================================

class AssessmentSession:
    """
    A stateful session for managing the medical assessment flow.
    Methods allow you to:
      - Retrieve the initial screening questions.
      - Submit responses.
      - Generate follow-up questions.
      - Generate examination recommendation questions.
      - Get final diagnosis and treatment recommendations.
    """

    def __init__(self):
        self.initial_responses: List[Dict[str, Any]] = []
        self.followup_responses: List[Dict[str, Any]] = []
        self.exam_responses: List[Dict[str, Any]] = []
        self.structured_questions: List[Dict[str, Any]] = []
        self.examination_history: List[Dict[str, Any]] = []

    def get_initial_questions(self) -> List[Dict[str, Any]]:
        """
        Return the predefined initial screening questions.
        """
        return questions_init

    def submit_initial_responses(self, responses: List[Dict[str, Any]]) -> None:
        """
        Save the initial responses provided by the user.
        (Validation may be performed in the frontend.)
        """
        self.initial_responses = responses

    def generate_followup_question(self) -> Dict[str, Any]:
        try:
            # Get the initial complaint using .get() for safety.
            initial_complaint = next(
                (resp.get('answer', '') for resp in self.initial_responses
                if resp.get('question', '') == "Please describe what brings you here today"),
                ""
            )
            context = f"Initial complaint: {initial_complaint}\n"
            
            # Build context from follow-up responses, using .get() to avoid KeyErrors.
            if self.followup_responses:
                context += "Previous responses:\n"
                for resp in self.followup_responses:
                    q_text = resp.get('question', 'N/A')
                    a_text = resp.get('answer', 'N/A')
                    context += f"Q: {q_text}\nA: {a_text}\n"
            
            # Get embedding and relevant documents.
            embedding = get_embedding_batch([context])[0]
            index = pc.Index("final-asha")
            relevant_docs = vectorQuotesWithSource(embedding, index)
            citations = relevant_docs[-5:] if relevant_docs else []
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]]) if relevant_docs else ""
            
            # Create a list of previously asked questions safely.
            previous_questions = (
                "\n".join([f"- {resp.get('question', 'N/A')}" for resp in self.followup_responses])
                if self.followup_responses else "No previous questions yet"
            )
            
            prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
                    
    Previous questions asked:
    {previous_questions}
                    
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
                max_tokens=150
            )
            question_text = completion.choices[0].message.content.strip()

            # Generate answer options for the question.
            options_prompt = f'''Generate 4 concise answers for: "{question_text}"
    Clear, mutually exclusive options.
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
                    # Remove leading numbering if present.
                    if text[0].isdigit() and text[1] in ['.', '-', ')']:
                        text = text[2:].strip()
                    options.append({"id": i + 1, "text": text})
            options.append({"id": 5, "text": "Other (please specify)"})

            # Process with MATRIX using the follow-up responses.
            matrix_output = process_with_matrix(question_text, self.followup_responses)

            return {
                "question": question_text,
                "options": options,
                "type": "MC",
                "citations": citations,
                "matrix_output": matrix_output
            }
        except Exception as e:
            return {"error": f"Error generating follow-up question: {str(e)}"}


    def submit_followup_response(self, response: Dict[str, Any]) -> None:
        """
        Submit a follow-up response. The response should be a dict that includes:
          - "question": The question text.
          - "answer": The user's answer.
          - "type": The question type.
          - Optionally, "matrix_output" and "citations".
        The structured version of the question is also stored.
        """
        self.followup_responses.append(response)
        if "matrix_output" in response and "options" in response:
            structured = parse_question_data(
                response["question"],
                response["options"],
                response["answer"],
                response["matrix_output"],
                response.get("citations", [])
            )
            self.structured_questions.append(structured)

    def should_stop_followup(self) -> bool:
        """
        Return True if sufficient follow-up information has been gathered.
        """
        if not self.followup_responses:
            return False
        last_question = self.followup_responses[-1].get("question", "")
        return judge(self.followup_responses, last_question)

    def generate_examination_question(self) -> Dict[str, Any]:
        """
        Generate a recommended examination (non-imaging, basic exam) based on the current responses.
        Returns a dict with keys:
          - "examination": The examination name and procedure.
          - "options": A list of possible findings.
          - "type": "EXAM".
          - "citations": Relevant citations.
        If the exam phase should be ended, returns {"phase_complete": True}.
        """
        try:
            initial_complaint = next(
                (resp['answer'] for resp in self.initial_responses
                 if resp['question'] == "Please describe what brings you here today"),
                ""
            )
            symptoms = set()
            for resp in self.followup_responses:
                answer = resp.get('answer', "").lower()
                for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling',
                                'headache', 'nausea', 'dizziness', 'rash']:
                    if symptom in answer:
                        symptoms.add(symptom)
            context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}
Previous exams: {json.dumps([exam.get('examination', '') for exam in self.exam_responses]) if self.exam_responses else "None"}"""
            embedding = get_embedding_batch([context])[0]
            index = pc.Index("final-asha")
            relevant_docs = vectorQuotesWithSource(embedding, index)
            exam_citations = relevant_docs
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]]) if relevant_docs else ""
            prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {json.dumps([exam.get('examination', '') for exam in self.exam_responses]) if self.exam_responses else "None"}

Recommend ONE essential examination in this format (avoid high‐tech exams like MRI, CT, etc.):
[Examination name and procedure]
#:[First possible finding]
#:[Second possible finding]
#:[Third possible finding]
#:[Fourth possible finding]'''
            completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=200
            )
            exam_text = completion.choices[0].message.content.strip()

            if judge_exam(self.exam_responses, exam_text):
                return {"phase_complete": True}

            examination, option_texts = parse_examination_text(exam_text)
            options = [{"id": i + 1, "text": text} for i, text in enumerate(option_texts)]
            options.append({"id": 5, "text": "Other (please specify)"})
            return {
                "examination": examination,
                "options": options,
                "type": "EXAM",
                "citations": exam_citations
            }
        except Exception as e:
            return {"error": f"Error generating examination question: {str(e)}"}

    def submit_examination_response(self, response: Dict[str, Any]) -> None:
        """
        Submit an examination response. The response should include:
          - "examination": The exam text.
          - "result": The chosen finding.
          - "type": "EXAM".
          - Optionally, "citations".
        The examination history is updated accordingly.
        """
        self.exam_responses.append(response)
        try:
            exam_text = response.get("examination", "")
            result = response.get("result", "")
            examination, options = parse_examination_text(exam_text)
            exam_entry = {
                "examination": examination,
                "options": options,
                "selected_option": result
            }
            self.examination_history.append(exam_entry)
        except Exception as e:
            self.examination_history.append(response)

    def get_final_diagnosis_and_treatment(self) -> Dict[str, Any]:
        """
        Get final diagnosis and treatment recommendations.
        Returns a dictionary with keys: "diagnosis", "treatment", and "citations".
        """
        return get_diagnosis_and_treatment(self.initial_responses, self.followup_responses, self.exam_responses)


def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                                followup_responses: List[Dict[str, Any]], 
                                exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get diagnosis and treatment recommendations with citations.
    This function makes several API calls to generate a diagnosis and various treatment parts.
    """
    try:
        initial_complaint = next(
            (resp['answer'] for resp in initial_responses 
             if resp['question'] == "Please describe what brings you here today"),
            ""
        )
        key_findings = []
        for resp in followup_responses + exam_responses:
            if isinstance(resp.get('answer'), str):
                key_findings.append(resp['answer'])
        key_findings = key_findings[-3:]  # use only the most recent findings

        # Get diagnosis using some context and citation look-up
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

        # Get treatment recommendations in parts
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

        # 2. Medications/Supplements
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

        treatment = "\n".join(treatment_parts)
        citations = diagnosis_docs + treatment_docs

        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": citations
        }

    except Exception as e:
        print(f"Error in diagnosis/treatment: {e}")
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
            print(f"Fallback also failed: {e2}")
            return {
                "diagnosis": "Error generating diagnosis",
                "treatment": "Error generating treatment",
                "citations": []
            }

# =============================================================================
# End of module
# =============================================================================
