from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_functions import (
    getIndex, get_embedding, vectorQuotes,
    process_chat_response, questions_init,
    ChatStage
)
import json
from typing import Dict, Any
import uuid

app = Flask(__name__)
CORS(app)

# Store conversation states
conversations: Dict[str, Dict[str, Any]] = {}

@app.route('/initial_questions', methods=['GET'])
def get_initial_questions():
    """Return initial screening questions."""
    return jsonify(questions_init)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate responses."""
    try:
        data = request.json
        message = data.get('message', '')
        screening_answers = data.get('screeningAnswers', [])
        conversation_id = data.get('conversationId')
        conversation_history = data.get('conversationHistory', [])
        current_stage = data.get('stage', ChatStage.INITIAL_QUESTIONS)

        print(f"Received message: {message}")
        print(f"Current stage: {current_stage}")
        
        # Create new conversation if needed
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                'screening_answers': screening_answers,
                'history': [],
                'stage': ChatStage.INITIAL_QUESTIONS
            }
            print(f"Created new conversation: {conversation_id}")
        
        # Get or update existing conversation
        conversation = conversations.get(conversation_id)
        if conversation:
            conversation['history'] = conversation_history
            conversation['stage'] = current_stage
        
        # Process the message
        context = {
            'screeningAnswers': screening_answers,
            'conversationHistory': conversation_history,
            'stage': current_stage
        }
        
        response = process_chat_response(
            message=message,
            context=context,
            stage=current_stage
        )
        
        # Update conversation stage
        if conversation:
            conversation['stage'] = response.get('stage', current_stage)
            print(f"Updated stage to: {conversation['stage']}")
        
        # Add conversation ID to response
        response['conversationId'] = conversation_id
        
        return jsonify(response)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "question": "Something went wrong. Please choose:",
            "type": "MC",
            "options": [
                {"id": 1, "text": "Try again"},
                {"id": 2, "text": "Start over"},
                {"id": 3, "text": "Contact support"}
            ],
            "stage": current_stage
        }), 500

@app.route('/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Retrieve conversation state."""
    try:
        conversation = conversations.get(conversation_id)
        if conversation:
            return jsonify(conversation)
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/conversation/<conversation_id>', methods=['DELETE'])
def end_conversation(conversation_id):
    """End and remove a conversation."""
    try:
        if conversation_id in conversations:
            del conversations[conversation_id]
            return jsonify({"message": "Conversation ended successfully"})
        return jsonify({"error": "Conversation not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "active_conversations": len(conversations)
    })

@app.before_request
def log_request_info():
    """Log incoming request details for debugging."""
    if request.method == 'POST':
        print(f"Endpoint: {request.endpoint}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Body: {request.get_json() if request.is_json else 'No JSON body'}")

@app.after_request
def log_response_info(response):
    """Log response details for debugging."""
    print(f"Response status: {response.status}")
    print(f"Response headers: {dict(response.headers)}")
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)