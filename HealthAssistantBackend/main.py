from flask import Flask, request, jsonify
from flask_cors import CORS
import ML

app = Flask(__name__)
CORS(app)

@app.route('/api/start-assessment', methods=['POST'])
def start_assessment():
    try:
        return jsonify({
            'status': 'success',
            'questions': ML.questions_init,
            'session_id': 1
        })
    except Exception as e:
        print(f"Error starting assessment: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/followup-questions', methods=['POST'])
def followup_questions():
    try:
        data = request.json
        initial_responses = data.get('initial_responses', [])
        followup_responses = data.get('followup_responses', [])

        # Generate next question using ML.py function
        next_question = ML.generate_question_with_options(initial_responses)
        
        # Check if we should move to examination phase
        is_examination_phase = ML.judge(followup_responses, next_question['question']) if followup_responses else False

        return jsonify({
            'status': 'success',
            'data': {
                'question': next_question['question'],
                'options': next_question['options'],
                'is_examination_phase': is_examination_phase
            }
        })
    except Exception as e:
        print(f"Error generating follow-up question: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/submit-assessment', methods=['POST'])
def submit_assessment():
    try:
        data = request.json
        return jsonify({
            'status': 'success',
            'message': 'Assessment data received successfully',
            'session_id': data.get('session_id')
        })
    except Exception as e:
        print(f"Error processing assessment data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)