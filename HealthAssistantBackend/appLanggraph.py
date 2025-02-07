import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from groq import Groq
from pinecone import Pinecone
import openai
import torch

# Import MATRIX components
from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

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

# Store conversation state with enhanced structure
conversations: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------

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
    """Search vector DB (Pinecone) and return relevant matches with source information."""
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

def call_groq_llm(prompt: str, model: str = "llama-3.3-70b-versatile", 
                 temperature: float = 0.3, max_tokens: int = 150) -> str:
    """Helper function to call Groq's LLM with error handling."""
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq LLM: {e}")
        return ""

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                       context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced error handling."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        
        state = matrix.state_encoder.encode_state(
            [],  # dummy history
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
                
            if len(prev_procedure) > 0 and len(current_procedure) > 0:
                words1 = set(prev_procedure.split())
                words2 = set(current_procedure.split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if similarity > 0.7:
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
        
        return should_end
                
    except Exception as e:
        print(f"Warning: Exam similarity check falling back to basic method: {e}")
        return len(previous_exams) >= MATRIXConfig.MAX_EXAMS

def parse_examination_text(examination_text: str) -> Tuple[str, List[str]]:
    """Parse examination text using "#:" delimiter."""
    parts = examination_text.split("#:")
    
    if len(parts) < 2:
        raise ValueError("Invalid examination format - missing '#:' delimiter")
        
    examination = parts[0].strip()
    options = [opt.strip() for opt in parts[1:] if opt.strip()]
    return examination, options

# --------------------------------------------------------------------
# FLASK ROUTES AND CORE LOGIC
# --------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/initial_questions", methods=["GET"])
def get_initial_questions():
    """Return initial screening questions."""
    return jsonify([
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
    ])

@app.route("/chat", methods=["POST"])
def chat():
    """Enhanced chat endpoint with robust error handling and state management."""
    try:
        data = request.json or {}
        message = data.get("message", "")
        conversation_id = data.get("conversationId")
        initial_responses = data.get("initialResponses", [])
        stage = data.get("stage", "followup")
        
        if initial_responses:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                "initial_responses": initial_responses,
                "followup_responses": [],
                "examination_responses": [],
                "structured_questions": [],
                "examination_history": [],
                "stage": "followup",
                "question_index": 0,
                "matrix_state": {"optimist_weight": 0.5, "pessimist_weight": 0.5}
            }

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                "initial_responses": [],
                "followup_responses": [],
                "examination_responses": [],
                "structured_questions": [],
                "examination_history": [],
                "stage": "initial",
                "question_index": 0,
                "matrix_state": {"optimist_weight": 0.5, "pessimist_weight": 0.5}
            }

        conversation = conversations[conversation_id]

        if conversation["stage"] == "initial" and conversation.get("initial_responses"):
            conversation["stage"] = "followup"
            return generate_followup_question(conversation, conversation_id)

        elif conversation["stage"] == "followup":
            if message:
                store_followup_response(conversation, message)

            if should_move_to_examinations(conversation):
                conversation["stage"] = "exam"
                return generate_examination(conversation, conversation_id)

            return generate_followup_question(conversation, conversation_id)

        elif conversation["stage"] == "exam":
            if message:
                store_examination_response(conversation, message)

            if should_move_to_diagnosis(conversation):
                conversation["stage"] = "complete"
                return generate_diagnosis(conversation, conversation_id)

            return generate_examination(conversation, conversation_id)

        return jsonify({"error": "Invalid conversation stage"})

    except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return jsonify({
            "error": str(e),
            "conversationId": conversation_id,
            "question": "An error occurred. Please choose:",
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        }), 500

def generate_followup_question(conversation: Dict, conversation_id: str) -> Dict:
    """Generate a follow-up question with vector search and MATRIX analysis."""
    try:
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        context = f"Initial complaint: {initial_complaint}\n"
        if conversation["followup_responses"]:
            context += "Previous responses:\n"
            for resp in conversation["followup_responses"]:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        embedding = get_embedding_batch([context])[0]
        index = pc.Index("final-asha")
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            raise ValueError("Could not find relevant medical context")
            
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        previous_questions = "\n".join([
            f"- {resp['question']}" 
            for resp in conversation["followup_responses"]
        ])
        
        prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
        
Previous questions asked:
{previous_questions if conversation["followup_responses"] else "No previous questions yet"}

Relevant medical context:
{combined_context}

Generate ONE focused, relevant follow-up question that is different from the previous questions.
Follow standard medical assessment order:
1. Duration and onset
2. Characteristics and severity
3. Associated symptoms
4. Impact on daily life

Return only the question text.'''
        
        question = call_groq_llm(prompt)
        conversation["current_question"] = question
        
        # Generate options
        options_prompt = f'''Generate 4 concise, clinically relevant answers for: "{question}"
Each option should be clear and mutually exclusive.
Return each option on a new line (1-4).'''
        
        options_text = call_groq_llm(options_prompt, temperature=0.2)
        options = []
        
        # Parse options, handling different number formats
        for i, opt in enumerate(options_text.split('\n')):
            if opt.strip():
                text = opt.strip()
                # Remove leading numbers and common separators
                if text[0].isdigit():
                    # Find where the actual text starts after number and separator
                    for j, char in enumerate(text[1:], 1):
                        if char not in ['.', '-', ')', ' ']:
                            text = text[j:].strip()
                            break
                options.append({"id": i + 1, "text": text})
        
        # Add "Other" option
        options.append({"id": 5, "text": "Other (please specify)"})
        
        # Get MATRIX analysis
        matrix_output = process_with_matrix(question, conversation["followup_responses"])
        
        return jsonify({
            "conversationId": conversation_id,
            "question": question,
            "type": "MC",
            "options": options,
            "matrixAnalysis": {
                "confidence": matrix_output.get("confidence", 0.5),
                "selectedAgent": matrix_output.get("selected_agent", "optimist"),
                "weights": matrix_output.get("weights", {"optimist": 0.5, "pessimist": 0.5})
            },
            "citations": relevant_docs[-5:]
        })
        
    except Exception as e:
        print(f"Error generating followup question: {e}")
        return jsonify({
            "error": str(e),
            "conversationId": conversation_id,
            "question": "An error occurred. Please choose:",
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

def generate_examination(conversation: Dict, conversation_id: str) -> Dict:
    """Generate examination recommendations with vector search and MATRIX analysis."""
    try:
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        # Track symptoms from followup responses
        symptoms = set()
        for resp in conversation["followup_responses"]:
            answer = resp["answer"].lower()
            for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                           'headache', 'nausea', 'dizziness', 'rash']:
                if symptom in answer:
                    symptoms.add(symptom)

        context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}
Previous findings: {str([exam['examination'] for exam in conversation['examination_responses']]) if conversation['examination_responses'] else "None"}"""

        embedding = get_embedding_batch([context])[0]
        index = pc.Index("final-asha")
        relevant_docs = vectorQuotesWithSource(embedding, index)
        
        if not relevant_docs:
            raise ValueError("Could not find relevant medical context")
            
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {str([exam['examination'] for exam in conversation['examination_responses']]) if conversation['examination_responses'] else "None"}

Recommend ONE essential examination in this format:
[Examination name and procedure]
#:[First possible finding]
#:[Second possible finding]
#:[Third possible finding]
#:[Fourth possible finding]'''

        examination_text = call_groq_llm(prompt, max_tokens=200)
        conversation["current_exam"] = examination_text
        
        try:
            examination, option_texts = parse_examination_text(examination_text)
            options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
            options.append({"id": 5, "text": "Other (please specify)"})
            
            matrix_output = process_with_matrix(examination_text, conversation["examination_responses"])
            
            return jsonify({
                "conversationId": conversation_id,
                "examination": examination,
                "type": "EXAM",
                "options": options,
                "matrixAnalysis": {
                    "confidence": matrix_output.get("confidence", 0.5),
                    "selectedAgent": matrix_output.get("selected_agent", "optimist"),
                    "weights": matrix_output.get("weights", {"optimist": 0.5, "pessimist": 0.5})
                },
                "citations": relevant_docs[-5:]
            })
            
        except ValueError as e:
            print(f"Error parsing examination: {e}")
            raise
            
    except Exception as e:
        print(f"Error generating examination: {e}")
        return jsonify({
            "error": str(e),
            "conversationId": conversation_id,
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

def generate_diagnosis(conversation: Dict, conversation_id: str) -> Dict:
    """Generate final diagnosis and treatment plan with comprehensive analysis."""
    try:
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        key_findings = []
        for resp in conversation["followup_responses"] + conversation["examination_responses"]:
            if isinstance(resp.get("answer"), str):
                key_findings.append(resp["answer"])
            elif isinstance(resp.get("result"), str):
                key_findings.append(resp["result"])
        key_findings = key_findings[-3:]  # Only keep last 3 findings

        index = pc.Index("final-asha")
        
        # Get relevant context for diagnosis
        diagnosis_embedding = get_embedding_batch([initial_complaint + " diagnosis"])[0]
        diagnosis_docs = vectorQuotesWithSource(diagnosis_embedding, index, top_k=2)
        
        diagnosis_context = " ".join([doc["text"] for doc in diagnosis_docs])
        diagnosis_prompt = f'''Patient complaint: {initial_complaint}
Key findings: {"; ".join(key_findings)}
Reference: {diagnosis_context[:200]}

List top 3-4 possible diagnoses based on symptoms.'''

        diagnosis = call_groq_llm(diagnosis_prompt, temperature=0.2, max_tokens=100)
        
        # Generate treatment plan in parts
        treatment_parts = []
        treatment_docs = []
        
        # Immediate Care
        immediate_embedding = get_embedding_batch([initial_complaint + " immediate care steps"])[0]
        immediate_docs = vectorQuotesWithSource(immediate_embedding, index, top_k=1)
        treatment_docs.extend(immediate_docs)
        if immediate_docs:
            immediate_prompt = f'''Based on: {immediate_docs[0]["text"][:200]}
Provide 2-3 immediate care steps for {initial_complaint}.'''
            immediate_response = call_groq_llm(immediate_prompt, temperature=0.2)
            treatment_parts.append("Immediate Care:\n" + immediate_response)
        
        # Medications
        med_embedding = get_embedding_batch([initial_complaint + " medications treatment"])[0]
        med_docs = vectorQuotesWithSource(med_embedding, index, top_k=1)
        treatment_docs.extend(med_docs)
        if med_docs:
            med_prompt = f'''Based on: {med_docs[0]["text"][:200]}
List 2-3 key medications or supplements for {initial_complaint}.'''
            med_response = call_groq_llm(med_prompt, temperature=0.2)
            treatment_parts.append("\nMedications/Supplements:\n" + med_response)
        
        # Home Care
        home_embedding = get_embedding_batch([initial_complaint + " home care follow up"])[0]
        home_docs = vectorQuotesWithSource(home_embedding, index, top_k=1)
        treatment_docs.extend(home_docs)
        if home_docs:
            home_prompt = f'''Based on: {home_docs[0]["text"][:200]}
List 2-3 home care instructions for {initial_complaint}.'''
            home_response = call_groq_llm(home_prompt, temperature=0.2)
            treatment_parts.append("\nHome Care:\n" + home_response)
        
        treatment = "\n".join(treatment_parts)
        
        # Compile all citations
        all_citations = diagnosis_docs + treatment_docs
        
        # Final MATRIX analysis
        matrix_output = process_with_matrix(
            diagnosis + "\n" + treatment,
            conversation["followup_responses"] + conversation["examination_responses"]
        )
        
        return jsonify({
            "conversationId": conversation_id,
            "type": "COMPLETE",
            "diagnosis": diagnosis,
            "treatment": treatment,
            "matrixAnalysis": {
                "confidence": matrix_output.get("confidence", 0.5),
                "selectedAgent": matrix_output.get("selected_agent", "optimist"),
                "weights": matrix_output.get("weights", {"optimist": 0.5, "pessimist": 0.5})
            },
            "citations": all_citations[-5:],
            "structuredQuestions": conversation.get("structured_questions", []),
            "examinationHistory": conversation.get("examination_history", [])
        })
            
    except Exception as e:
        print(f"Error generating diagnosis: {e}")
        return jsonify({
            "error": str(e),
            "conversationId": conversation_id,
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

def store_followup_response(conversation: Dict, message: str):
    """Store followup response with enhanced tracking."""
    current_question = conversation.get("current_question", "")
    matrix_output = process_with_matrix(current_question, conversation["followup_responses"])
    
    response = {
        "question": current_question,
        "answer": message,
        "type": "MC",
        "matrix_analysis": matrix_output
    }
    
    conversation["followup_responses"].append(response)
    conversation["matrix_state"] = matrix_output.get("weights", 
                                                   {"optimist": 0.5, "pessimist": 0.5})

def store_examination_response(conversation: Dict, message: str):
    """Store examination response with enhanced tracking."""
    current_exam = conversation.get("current_exam", "")
    matrix_output = process_with_matrix(current_exam, conversation["examination_responses"])
    
    response = {
        "examination": current_exam,
        "result": message,
        "type": "EXAM",
        "matrix_analysis": matrix_output
    }
    
    conversation["examination_responses"].append(response)
    conversation["matrix_state"].update(matrix_output.get("weights", {}))

def should_move_to_examinations(conversation: Dict) -> bool:
    """Determine if we should move to examinations phase."""
    if len(conversation["followup_responses"]) < 3:
        return False
        
    matrix_state = conversation["matrix_state"]
    return (
        matrix_state.get("optimist_weight", 0.5) > 0.7 or
        len(conversation["followup_responses"]) >= MATRIXConfig.MAX_FOLLOWUPS
    )

def should_move_to_diagnosis(conversation: Dict) -> bool:
    """Determine if we should move to diagnosis phase."""
    if len(conversation["examination_responses"]) < 2:
        return False
        
    matrix_state = conversation["matrix_state"]
    return (
        matrix_state.get("optimist_weight", 0.5) > 0.8 or
        len(conversation["examination_responses"]) >= MATRIXConfig.MAX_EXAMS
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)