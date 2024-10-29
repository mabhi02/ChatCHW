import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initial screening questions
questions_init = [
    {
        "question": "What is the patient's sex?",
        "type": "MC",
        "options": [
            {"id": 1, "text": "Male"},
            {"id": 2, "text": "Female"},
            {"id": 3, "text": "Non-binary"},
            {"id": 4, "text": "Other"}
        ]
    },
    {
        "question": "What is the patient's age?",
        "type": "NUM",
        "options": [],
        "range": {
            "min": 0,
            "max": 120,
            "step": 1,
            "unit": "years"
        }
    },
    {
        "question": "Does the patient have a caregiver?",
        "type": "YN",
        "options": [
            {"id": "yes", "text": "Yes"},
            {"id": "no", "text": "No"},
            {"id": "not_sure", "text": "Not sure"}
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
            {"id": 5, "text": "Other"}
        ]
    },
    {
        "question": "What brings the patient here?",
        "type": "MCM",  # Changed from FREE to MCM
        "options": [
            {"id": 1, "text": "Fever"},
            {"id": 2, "text": "Cough"},
            {"id": 3, "text": "Headache"},
            {"id": 4, "text": "Body Pain"},
            {"id": 5, "text": "Stomach Issues"},
            {"id": 6, "text": "Breathing Problems"},
            {"id": 7, "text": "Skin Problems"},
            {"id": 8, "text": "Other"}
        ]
    }
]

# Helper function to validate and process multiple choice inputs
def process_input(input_data, question_type, options=None):
    if question_type == "MC":
        choice_id = input_data
        if any(opt["id"] == choice_id for opt in options):
            return next(opt["text"] for opt in options if opt["id"] == choice_id)
        raise ValueError("Invalid choice for MC question")
    
    elif question_type == "NUM":
        value = int(input_data)
        if 0 <= value <= 120:
            return str(value)
        raise ValueError("Invalid age value")
    
    elif question_type == "YN":
        if input_data in ["yes", "no", "not_sure"]:
            return next(opt["text"] for opt in options if opt["id"] == input_data)
        raise ValueError("Invalid choice for YN question")
    
    elif question_type == "MCM":
        if isinstance(input_data, list):
            choice_ids = input_data
            if all(any(opt["id"] == id for opt in options) for id in choice_ids):
                selected_texts = [opt["text"] for opt in options if opt["id"] in choice_ids]
                return ", ".join(selected_texts)
        raise ValueError("Invalid choices for MCM question")
    
    return input_data

def getIndex():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("final-asha")
    return index

def get_embedding(text):
    response = openai.Embedding.create(input=text, engine="text-embedding-3-small")
    return response['data'][0]['embedding']

def getRes(query_embedding, index):
    res = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return res

def vectorQuotes(query_embedding, index):
    similarity = getRes(query_embedding, index)
    return [{"text": match['metadata']['text'], "id": match['id']} for match in similarity['matches']]

def find_most_relevant_quote(query_embedding, quotes):
    quote_embeddings = [get_embedding(quote['text']) for quote in quotes]
    similarities = cosine_similarity([query_embedding], quote_embeddings)[0]
    most_relevant_index = np.argmax(similarities)
    return quotes[most_relevant_index]

def groqCall(prompt):
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama3-70b-8192",
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        return None

def generate_followup_question(main_symptom, screening_info, previous_questions, relevant_quote):
    prompt = f"""Generate a single, concise follow-up question about: "{main_symptom}"
    Consider this screening information: {screening_info}
    The question should be different from these previous questions: {previous_questions}
    Base your question on this relevant information: {relevant_quote['text']}
    Format the response as a multiple choice question with 4 options.
    Format:
    Question: [Your question here]
    Options:
    1. [Option 1]
    2. [Option 2]
    3. [Option 3]
    4. [Option 4]"""
    return groqCall(prompt)

def generate_test(main_symptom, screening_info, symptom_info, relevant_quote, previous_tests):
    prompt = f"""Suggest a simple test for: "{main_symptom}"
    Consider this screening information: {screening_info}
    And these symptoms: {symptom_info}
    The test should be different from these previous tests: {previous_tests}
    Base your suggestion on this relevant information: {relevant_quote['text']}
    Format the response as a test with multiple choice results.
    Format:
    Test: [Test Name]
    Description: [Brief description of the test]
    Instructions:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    Results:
    1. Normal
    2. Mild
    3. Moderate
    4. Severe"""
    return groqCall(prompt)

def generate_advice(main_symptom, screening_info, symptom_info, test_results, relevant_quote):
    prompt = f"""Provide 3 pieces of advice for: "{main_symptom}"
    Consider this screening information: {screening_info}
    These symptoms: {symptom_info}
    And these test results: {test_results}
    Base your advice on this relevant information: {relevant_quote['text']}
    Provide the 3 pieces of advice, each followed by a brief summary of the citation used.
    Format your response as follows:
    1. [Advice 1]
    Citation summary: [Brief summary of the citation used for Advice 1]
    
    2. [Advice 2]
    Citation summary: [Brief summary of the citation used for Advice 2]
    
    3. [Advice 3]
    Citation summary: [Brief summary of the citation used for Advice 3]"""
    return groqCall(prompt)

def generate_conversation_followup(conversation_summary):
    prompt = f"""Based on this conversation summary: {conversation_summary}
    Generate a follow-up question or suggestion to continue the conversation.
    Format the response as a multiple choice question with 4 options.
    Format:
    Question: [Your question here]
    Options:
    1. [Option 1]
    2. [Option 2]
    3. [Option 3]
    4. [Option 4]"""
    return groqCall(prompt)

@app.route('/initial_questions', methods=['GET'])
def get_initial_questions():
    return jsonify(questions_init)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']  # This will now be the selected option ID(s)
    question_type = data.get('question_type', 'MCM')  # Default to MCM if not specified
    options = data.get('options', [])
    screening_answers = data.get('screening_answers', [])
    conversation_history = data.get('conversation_history', [])
    
    try:
        # Process the input based on the question type
        processed_message = process_input(message, question_type, options)
        
        index = getIndex()
        symptom_embedding = get_embedding(processed_message)
        relevant_quotes = vectorQuotes(symptom_embedding, index)
        
        screening_info = ", ".join([f"{a['question']}: {a['answer']}" for a in screening_answers])
        
        if len(conversation_history) < 3:
            previous_questions = [q['question'] for q in conversation_history]
            response = generate_followup_question(processed_message, screening_info, previous_questions, relevant_quotes[0])
            return jsonify({
                "response": response,
                "type": "followup_question",
                "format": "multiple_choice"
            })
        
        elif len(conversation_history) == 3:
            symptom_info = [f"Symptom: {q['question']} Answer: {q['answer']}" for q in conversation_history]
            test = generate_test(processed_message, screening_info, symptom_info, relevant_quotes[0], [])
            return jsonify({
                "response": test,
                "type": "test",
                "format": "multiple_choice"
            })
        
        elif len(conversation_history) == 4:
            symptom_info = [f"Symptom: {q['question']} Answer: {q['answer']}" for q in conversation_history[:3]]
            test_result = conversation_history[3]['answer']
            advice = generate_advice(processed_message, screening_info, symptom_info, [test_result], relevant_quotes[0])
            return jsonify({
                "response": advice,
                "type": "advice"
            })
        
        else:
            conversation_summary = f"Main symptom: {processed_message}, Screening info: {screening_info}, " \
                                   f"Conversation history: {conversation_history}"
            followup = generate_conversation_followup(conversation_summary)
            return jsonify({
                "response": followup,
                "type": "conversation_followup",
                "format": "multiple_choice"
            })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)