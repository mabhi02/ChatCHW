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
        "type": "FREE",
        "options": []
    }
]

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
    Provide only the question, nothing else."""
    return groqCall(prompt)

def generate_test(main_symptom, screening_info, symptom_info, relevant_quote, previous_tests):
    prompt = f"""Suggest a simple test for: "{main_symptom}"
    Consider this screening information: {screening_info}
    And these symptoms: {symptom_info}
    The test should be different from these previous tests: {previous_tests}
    Base your suggestion on this relevant information: {relevant_quote['text']}
    Provide the test description and step-by-step instructions on how to perform it.
    Format your response as follows:
    Test: [Test Name]
    Description: [Brief description of the test]
    Instructions:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    ...
    """
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
    Citation summary: [Brief summary of the citation used for Advice 3]
    """
    return groqCall(prompt)

def generate_conversation_followup(conversation_summary):
    prompt = f"""Based on this conversation summary: {conversation_summary}
    Generate a follow-up question or suggestion to continue the conversation.
    This could be asking for more details about a symptom, suggesting a referral, or recommending a follow-up appointment.
    Provide only the follow-up question or suggestion, nothing else."""
    return groqCall(prompt)

@app.route('/initial_questions', methods=['GET'])
def get_initial_questions():
    return jsonify(questions_init)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    screening_answers = data.get('screening_answers', [])
    conversation_history = data.get('conversation_history', [])
    
    try:
        index = getIndex()
        symptom_embedding = get_embedding(message)
        relevant_quotes = vectorQuotes(symptom_embedding, index)
        
        screening_info = ", ".join([f"{a['question']}: {a['answer']}" for a in screening_answers])
        
        # Determine the stage of the conversation
        if len(conversation_history) < 3:
            # Follow-up questions stage
            previous_questions = [q['question'] for q in conversation_history]
            response = generate_followup_question(message, screening_info, previous_questions, relevant_quotes[0])
            return jsonify({"response": response, "type": "followup_question"})
        
        elif len(conversation_history) == 3:
            # Test stage
            symptom_info = [f"Symptom: {q['question']} Answer: {q['answer']}" for q in conversation_history]
            test = generate_test(message, screening_info, symptom_info, relevant_quotes[0], [])
            return jsonify({"response": test, "type": "test"})
        
        elif len(conversation_history) == 4:
            # Advice stage
            symptom_info = [f"Symptom: {q['question']} Answer: {q['answer']}" for q in conversation_history[:3]]
            test_result = conversation_history[3]['answer']
            advice = generate_advice(message, screening_info, symptom_info, [test_result], relevant_quotes[0])
            return jsonify({"response": advice, "type": "advice"})
        
        else:
            # Conversation follow-up stage
            conversation_summary = f"Main symptom: {message}, Screening info: {screening_info}, " \
                                   f"Conversation history: {conversation_history}"
            followup = generate_conversation_followup(conversation_summary)
            return jsonify({"response": followup, "type": "conversation_followup"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)