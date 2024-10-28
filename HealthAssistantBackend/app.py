import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId, json_util
import json
from datetime import datetime
from ml_functions import (
    getIndex, get_embedding, vectorQuotes, find_most_relevant_quote,
    generate_followup_question, generate_test, generate_advice,
    generate_conversation_followup, questions_init
)

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["health_assistant"]
conversations_collection = db["conversations"]

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

@app.route('/', methods=['GET'])
def home():
    return "Health Assistant API is running"

@app.route('/initial_questions', methods=['GET'])
def get_initial_questions():
    return jsonify(questions_init)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    screening_answers = data.get('screeningAnswers', [])
    conversation_history = data.get('conversationHistory', [])
    conversation_id = data.get('conversationId')
    stage = data.get('stage', 'followup')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        index = getIndex()
        symptom_embedding = get_embedding(message)
        relevant_quotes = vectorQuotes(symptom_embedding, index)
        
        screening_info = ", ".join([f"{a.get('question', 'Unknown')}: {a.get('answer', 'No answer')}" for a in screening_answers])
        
        if stage == 'followup':
            previous_questions = [q['text'] for q in conversation_history if q['sender'] == 'bot']
            response = generate_followup_question(message, screening_info, previous_questions, relevant_quotes[0])
        elif stage == 'test':
            symptom_info = [f"Symptom: {q['text']}" for q in conversation_history if q['sender'] == 'user']
            response = generate_test(message, screening_info, symptom_info, relevant_quotes[0], [])
        elif stage == 'advice':
            symptom_info = [f"Symptom: {q['text']}" for q in conversation_history[:3] if q['sender'] == 'user']
            test_result = conversation_history[-1]['text']
            response = generate_advice(message, screening_info, symptom_info, [test_result], relevant_quotes[0])
        else:
            conversation_summary = f"Main symptom: {message}, Screening info: {screening_info}, " \
                                   f"Conversation history: {conversation_history}"
            response = generate_conversation_followup(conversation_summary)

        # Update conversation history
        conversation_history.append({"text": message, "sender": "user"})
        conversation_history.append({"text": response, "sender": "bot"})

        # Save conversation to MongoDB
        conversation_data = {
            "screening_answers": screening_answers,
            "conversation_history": conversation_history,
            "main_symptom": message,
            "stage": stage,
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if conversation_id:
            conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": conversation_data},
                upsert=True
            )
        else:
            result = conversations_collection.insert_one(conversation_data)
            conversation_id = str(result.inserted_id)

        return jsonify({"response": response, "conversationId": str(conversation_id)})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/full_conversation/<conversation_id>', methods=['GET'])
def get_full_conversation(conversation_id):
    try:
        conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify({"messages": conversation['conversation_history']})
    except Exception as e:
        app.logger.error(f"An error occurred fetching full conversation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/log', methods=['GET'])
def get_logs():
    page = int(request.args.get('page', 1))
    per_page = 10

    try:
        # Get total count
        total = conversations_collection.count_documents({})

        # Implement pagination
        logs = conversations_collection.find().sort("_id", -1).skip((page - 1) * per_page).limit(per_page)

        # Convert the MongoDB documents to a JSON serializable format
        logs_data = json.loads(json_util.dumps(list(logs)))

        return render_template(
            'logs_template.html',
            logs=logs_data,
            total=total,
            page=page,
            per_page=per_page
        )
    except Exception as e:
        app.logger.error(f"An error occurred fetching logs: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/log/<conversation_id>', methods=['GET'])
def get_specific_log(conversation_id):
    try:
        # Ensure the conversation_id is a valid ObjectId
        if not ObjectId.is_valid(conversation_id):
            return "Invalid conversation ID", 400

        conversation_data = conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if conversation_data:
            # Convert the MongoDB document to a JSON serializable format
            conversation_data = json.loads(json_util.dumps(conversation_data))
            # Convert the _id to a string
            conversation_data['_id'] = str(conversation_data['_id'])
            return render_template('conversation_log_template.html', conversation_data=conversation_data)
        else:
            return "Log not found", 404
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)