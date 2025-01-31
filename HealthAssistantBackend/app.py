import os
import uuid
from typing import Dict, List, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import all the necessary libraries from cmdML
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

# Store conversation state
conversations: Dict[str, Dict[str, Any]] = {}

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/initial_questions", methods=["GET"])
def get_initial_questions():
    """Return initial screening questions from cmdML."""
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
    """Main chat endpoint that handles the conversation flow."""
    try:
        data = request.json or {}
        message = data.get("message", "")
        conversation_id = data.get("conversationId")
        initial_responses = data.get("initialResponses", [])
        stage = data.get("stage", "followup")  # Default to followup if we have initial responses
        
        # If we have initial responses, store them and move to followup
        if initial_responses:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                "initial_responses": initial_responses,
                "followup_responses": [],
                "examination_responses": [],
                "stage": "followup",
                "question_index": 0
            }

        # Initialize new conversation if needed
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                "initial_responses": [],
                "followup_responses": [],
                "examination_responses": [],
                "stage": "initial",
                "question_index": 0
            }

        conversation = conversations[conversation_id]

        # Skip initial questions if we already have them
        if conversation["stage"] == "initial" and conversation.get("initial_responses"):
            conversation["stage"] = "followup"
            return generate_followup_question(conversation)

        # Handle followup questions stage
        elif conversation["stage"] == "followup":
            if message:
                # Store followup response
                conversation["followup_responses"].append({
                    "question": conversation.get("current_question", ""),
                    "answer": message,
                    "type": "MC"
                })

            # Check if we should move to examinations
            if len(conversation["followup_responses"]) >= 3:
                conversation["stage"] = "exam"
                return generate_examination(conversation)

            # Generate next followup question
            return generate_followup_question(conversation)

        # Handle examination stage
        elif conversation["stage"] == "exam":
            if message:
                # Store examination response
                conversation["examination_responses"].append({
                    "examination": conversation.get("current_exam", ""),
                    "result": message,
                    "type": "EXAM"
                })

            # Check if we should move to diagnosis
            if len(conversation["examination_responses"]) >= 2:
                conversation["stage"] = "complete"
                return generate_diagnosis(conversation)

            # Generate next examination
            return generate_examination(conversation)

        return jsonify({"error": "Invalid conversation stage"})

    except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return jsonify({
            "error": str(e),
            "question": "An error occurred. Please choose:",
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        }), 500

def generate_followup_question(conversation: Dict) -> Dict:
    """Generate a follow-up question using cmdML's logic."""
    try:
        # Get initial complaint
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        # Use groq to generate question
        context = f'''Initial complaint: {initial_complaint}
Previous responses: {str(conversation["followup_responses"])}'''

        question_prompt = f'''Based on the medical context: "{context}"
Generate ONE focused follow-up question following standard medical assessment:
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
        conversation["current_question"] = question

        # Generate options
        options_prompt = f'''Generate 4 concise, clinically relevant answers for: "{question}"
Each option should be clear and mutually exclusive.
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

        return jsonify({
            "question": question,
            "type": "MC",
            "options": options
        })

    except Exception as e:
        print(f"Error generating followup: {e}")
        return jsonify({
            "error": str(e),
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

def generate_examination(conversation: Dict) -> Dict:
    """Generate examination recommendations using cmdML's logic."""
    try:
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        # Generate exam based on complaint
        prompt = f'''Based on the patient's complaint: "{initial_complaint}"
Generate ONE essential medical examination recommendation:
1. Address reported symptoms directly
2. Use basic medical equipment
3. Be specific and detailed

Format:
Examination: [name]
Procedure: [detailed steps in a numbered list]'''

        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200
        )

        examination = completion.choices[0].message.content.strip()
        conversation["current_exam"] = examination

        # Generate findings options
        findings_prompt = f'''For the following examination:
{examination}

Generate 4 possible clinical findings that could result from this examination.
Include both normal and abnormal results.'''

        findings_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": findings_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=200
        )

        options = []
        findings = findings_completion.choices[0].message.content.strip().split('\n')
        for i, finding in enumerate(findings):
            if finding.strip():
                text = finding.strip()
                if text[0].isdigit() and text[1] in ['.', '-', ')']:
                    text = text[2:].strip()
                options.append({"id": i + 1, "text": text})

        options.append({"id": 5, "text": "Other (please specify)"})

        return jsonify({
            "examination": examination,
            "type": "EXAM",
            "options": options
        })

    except Exception as e:
        print(f"Error generating examination: {e}")
        return jsonify({
            "error": str(e),
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

def generate_diagnosis(conversation: Dict) -> Dict:
    """Generate final diagnosis and treatment plan using cmdML's logic."""
    try:
        initial_complaint = next((
            resp["answer"] for resp in conversation["initial_responses"]
            if resp["question"] == "Please describe what brings you here today"
        ), "")

        findings = []
        for resp in conversation["followup_responses"] + conversation["examination_responses"]:
            if isinstance(resp.get("answer"), str):
                findings.append(resp["answer"])
            elif isinstance(resp.get("result"), str):
                findings.append(resp["result"])

        diagnosis_prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key findings: {'; '.join(findings[-3:])}

Generate a concise clinical assessment including:
1. Top 2-3 possible diagnoses
2. Reasoning for each
3. Level of certainty

Format as a clear medical assessment.'''

        diagnosis_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": diagnosis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=200
        )

        diagnosis = diagnosis_completion.choices[0].message.content.strip()

        treatment_prompt = f'''Based on the assessment:
{diagnosis}

Provide a treatment plan including:
1. Immediate care steps
2. Medications/supplements if needed
3. Home care instructions
4. Follow-up recommendations
5. When to seek emergency care

Format in clear sections.'''

        treatment_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": treatment_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=300
        )

        treatment = treatment_completion.choices[0].message.content.strip()

        return jsonify({
            "diagnosis": diagnosis,
            "treatment": treatment,
            "type": "COMPLETE"
        })

    except Exception as e:
        print(f"Error generating diagnosis: {e}")
        return jsonify({
            "error": str(e),
            "type": "MC",
            "options": [
                {"id": "retry", "text": "Try again"},
                {"id": "restart", "text": "Start over"}
            ]
        })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)