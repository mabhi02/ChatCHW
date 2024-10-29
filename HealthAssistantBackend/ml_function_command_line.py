import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize environment and API clients
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
        "type": "MC",
        "options": [
            {"id": 1, "text": "Yes"},
            {"id": 2, "text": "No"},
            {"id": 3, "text": "Not sure"}
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

# RAG Functions
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
        print(f"\n❌ Error calling Groq API: {str(e)}")
        return None

# Input Handling Functions
def get_valid_input(question_type, options=None, range_info=None):
    """Generic input handler with validation"""
    while True:
        try:
            if question_type == "MC":
                print("\nOptions:")
                for opt in options:
                    print(f"{opt['id']}. {opt['text']}")
                choice = int(input("\nEnter option number: "))
                if any(opt["id"] == choice for opt in options):
                    selected_text = next(opt["text"] for opt in options if opt["id"] == choice)
                    print(f"\n✓ Selected: {selected_text}")
                    return selected_text
                print("\n❌ Invalid choice. Please select from the available options.")
            
            elif question_type == "NUM":
                value = int(input(f"\nEnter a number between {range_info['min']} and {range_info['max']} {range_info['unit']}: "))
                if range_info['min'] <= value <= range_info['max']:
                    print(f"\n✓ Recorded: {value} {range_info['unit']}")
                    return str(value)
                print(f"\n❌ Please enter a number between {range_info['min']} and {range_info['max']}")
            
            elif question_type == "MCM":
                print("\nOptions:")
                for opt in options:
                    print(f"{opt['id']}. {opt['text']}")
                choices = input("\nEnter option numbers, separated by commas: ")
                chosen_ids = [int(x.strip()) for x in choices.split(',')]
                valid_choices = [opt for opt in options if opt['id'] in chosen_ids]
                
                if len(valid_choices) == len(chosen_ids):
                    result = ", ".join(opt['text'] for opt in valid_choices)
                    print(f"\n✓ Selected: {result}")
                    return result
                print("\n❌ Invalid choice(s). Please select from the available options.")
            
            elif question_type == "FREE":
                response = input("\nPlease describe the patient's symptoms: ")
                if response.strip():
                    print(f"\n✓ Recorded: {response}")
                    return response
                print("\n❌ Please provide a description of the symptoms.")
            
        except ValueError:
            print("\n❌ Please enter a valid number.")

# Test Related Functions
def generate_test_options(test_type):
    """Generate appropriate options based on test type"""
    test_options = {
        "temperature": [
            {"id": 1, "text": "Normal (36.5-37.5°C)"},
            {"id": 2, "text": "Mild Fever (37.6-38.3°C)"},
            {"id": 3, "text": "High Fever (38.4-39.4°C)"},
            {"id": 4, "text": "Very High Fever (>39.5°C)"},
            {"id": 5, "text": "Custom..."}
        ],
        "blood_pressure": [
            {"id": 1, "text": "Normal (<120/80)"},
            {"id": 2, "text": "Elevated (120-129/80)"},
            {"id": 3, "text": "High Stage 1 (130-139/80-89)"},
            {"id": 4, "text": "High Stage 2 (≥140/90)"},
            {"id": 5, "text": "Custom..."}
        ],
        "pulse": [
            {"id": 1, "text": "Normal (60-100 bpm)"},
            {"id": 2, "text": "Bradycardia (<60 bpm)"},
            {"id": 3, "text": "Mild Tachycardia (100-120 bpm)"},
            {"id": 4, "text": "Severe Tachycardia (>120 bpm)"},
            {"id": 5, "text": "Custom..."}
        ],
        "respiratory_rate": [
            {"id": 1, "text": "Normal (12-20 breaths/min)"},
            {"id": 2, "text": "Mild Elevation (21-24)"},
            {"id": 3, "text": "Moderate Elevation (25-30)"},
            {"id": 4, "text": "Severe Elevation (>30)"},
            {"id": 5, "text": "Custom..."}
        ],
        "oxygen_saturation": [
            {"id": 1, "text": "Normal (95-100%)"},
            {"id": 2, "text": "Mild Hypoxemia (90-94%)"},
            {"id": 3, "text": "Moderate Hypoxemia (85-89%)"},
            {"id": 4, "text": "Severe Hypoxemia (<85%)"},
            {"id": 5, "text": "Custom..."}
        ],
        "stool_blood": [
            {"id": 1, "text": "No Blood"},
            {"id": 2, "text": "Trace Amount"},
            {"id": 3, "text": "Moderate Amount"},
            {"id": 4, "text": "Large Amount"},
            {"id": 5, "text": "Custom..."}
        ],
        "pain_scale": [
            {"id": 1, "text": "No Pain (0)"},
            {"id": 2, "text": "Mild Pain (1-3)"},
            {"id": 3, "text": "Moderate Pain (4-6)"},
            {"id": 4, "text": "Severe Pain (7-10)"},
            {"id": 5, "text": "Custom..."}
        ],
        "consciousness": [
            {"id": 1, "text": "Alert"},
            {"id": 2, "text": "Voice Responsive"},
            {"id": 3, "text": "Pain Responsive"},
            {"id": 4, "text": "Unresponsive"},
            {"id": 5, "text": "Custom..."}
        ],
        "default": [
            {"id": 1, "text": "Normal"},
            {"id": 2, "text": "Mild Abnormality"},
            {"id": 3, "text": "Moderate Abnormality"},
            {"id": 4, "text": "Severe Abnormality"},
            {"id": 5, "text": "Custom..."}
        ]
    }
    
    return test_options.get(test_type, test_options["default"])

def determine_test_type(test_name, description):
    """Determine the appropriate test type based on test name and description"""
    test_name = test_name.lower()
    description = description.lower()
    
    # Dictionary mapping keywords to test types
    test_keywords = {
        "temperature": ["temperature", "fever", "temp"],
        "blood_pressure": ["blood pressure", "bp", "pressure"],
        "pulse": ["pulse", "heart rate", "heartbeat"],
        "respiratory_rate": ["breathing", "respiratory", "breath rate"],
        "oxygen_saturation": ["oxygen", "o2", "saturation"],
        "stool_blood": ["stool", "blood in stool", "fecal"],
        "pain_scale": ["pain", "discomfort", "ache"],
        "consciousness": ["conscious", "awareness", "alert"]
    }
    
    for test_type, keywords in test_keywords.items():
        if any(word in test_name or word in description for word in keywords):
            return test_type
            
    return "default"

def determine_if_step_needs_measurement(step_text):
    """Determine if a step requires measurement/assessment input"""
    step_lower = step_text.lower()
    
    # Keywords indicating measurement/assessment is needed
    assessment_keywords = [
        "measure", "check", "assess", "record", "observe", "read", "count",
        "test result", "finding", "level", "rate", "pressure", "temperature",
        "saturation", "reading", "value", "score", "severity", "intensity"
    ]
    
    # Keywords indicating procedural steps (no measurement needed)
    procedural_keywords = [
        "wait", "prepare", "clean", "collect", "place", "position", "ask",
        "tell", "instruct", "help", "assist", "allow", "let", "ensure",
        "make sure", "verify", "sterilize", "wash", "dispose"
    ]
    
    # Check if step is a measurement/assessment step
    if any(keyword in step_lower for keyword in assessment_keywords):
        return True
    if any(keyword in step_lower for keyword in procedural_keywords):
        return False
        
    # Default to True for unrecognized steps to ensure we don't miss important measurements
    return True

def perform_test_step(step_text, test_type):
    """Modified test step handler that differentiates between measurement and procedural steps"""
    print(f"\nAssessing: {step_text}")
    
    # Check if this step needs measurement
    if not determine_if_step_needs_measurement(step_text):
        input("\nPress Enter when this step is completed...")
        return "Completed"
    
    # For steps requiring measurement, use the MCQ system
    options = generate_test_options(test_type)
    print("\nOptions:")
    for opt in options:
        print(f"{opt['id']}. {opt['text']}")
    
    while True:
        try:
            choice = int(input("\nEnter option number: "))
            if 1 <= choice <= 5:
                if choice == 5:  # Custom option
                    custom_value = input("\nEnter custom value: ")
                    print(f"\n✓ Recorded: {custom_value}")
                    return custom_value
                else:
                    result = options[choice-1]['text']
                    print(f"\n✓ Recorded: {result}")
                    return result
            print("\n❌ Please enter a number between 1 and 5.")
        except ValueError:
            print("\n❌ Please enter a valid number.")

def perform_test_step(step_text, test_type):
    """Modified test step handler that strictly respects PROCEDURAL label"""
    print(f"\nAssessing: {step_text}")
    
    # Check if step is explicitly marked as procedural
    if "(PROCEDURAL)" in step_text.upper():
        input("\nPress Enter once this step is completed...")
        return "Completed"
    
    # For measurement steps, use the MCQ system
    options = generate_test_options(test_type)
    print("\nOptions:")
    for opt in options:
        print(f"{opt['id']}. {opt['text']}")
    
    while True:
        try:
            choice = int(input("\nEnter option number: "))
            if 1 <= choice <= 5:
                if choice == 5:  # Custom option
                    custom_value = input("\nEnter custom value: ")
                    print(f"\n✓ Recorded: {custom_value}")
                    return custom_value
                else:
                    result = options[choice-1]['text']
                    print(f"\n✓ Recorded: {result}")
                    return result
            print("\n❌ Please enter a number between 1 and 5.")
        except ValueError:
            print("\n❌ Please enter a valid number.")

def generate_test(main_symptom, screening_info, symptom_info, relevant_quote, previous_tests):
    """Modified test generation to explicitly mark procedural steps"""
    prompt = f"""Suggest a simple test for: "{main_symptom}"
    Consider this screening information: {screening_info}
    And these symptoms: {symptom_info}
    The test should be different from these previous tests: {previous_tests}
    Base your suggestion on this relevant information: {relevant_quote['text']}
    
    For each step, mark procedural steps with (PROCEDURAL) at the end.
    Only steps that require measuring or assessing a value should not be marked as procedural.
    
    Format your response as follows:
    Test: [Test Name]
    Description: [Brief description of the test]
    Steps:
    1. Prepare testing materials and ensure clean workspace (PROCEDURAL)
    2. Measure the patient's vital signs
    3. Collect the required sample (PROCEDURAL)
    4. Record the test result reading"""
    
    return groqCall(prompt)

def perform_tests(main_symptom, screening_info, symptoms, symptom_embedding, relevant_quotes):
    """Conduct diagnostic tests with dynamic options"""
    tests = []
    print("\n=== Diagnostic Tests ===")
    symptom_info = [f"Symptom: {s['question']} Answer: {s['answer']}" for s in symptoms]
    
    for i in range(2):
        print(f"\nTest {i+1}/2")
        test_quote = find_most_relevant_quote(symptom_embedding, relevant_quotes)
        test_info = generate_test(
            main_symptom,
            screening_info,
            symptom_info,
            test_quote,
            [t['test_name'] for t in tests]
        )
        
        if test_info:
            test_lines = test_info.split('\n')
            test_name = test_lines[0].replace('Test: ', '').strip()
            test_description = test_lines[1].replace('Description: ', '').strip()
            
            # Determine test type
            test_type = determine_test_type(test_name, test_description)
            
            print(f"\nTest: {test_name}")
            print(f"Description: {test_description}")
            print("\nSteps:")
            
            instructions = []
            results = []
            for line in test_lines[3:]:
                if line.strip() and line[0].isdigit():
                    instructions.append(line)
                    result = perform_test_step(line.split('.', 1)[1].strip(), test_type)
                    results.append(result)
            
            tests.append({
                "test_name": test_name,
                "test_description": test_description,
                "instructions": instructions,
                "results": results,
                "relevant_quote": {"text": test_quote['text'], "id": test_quote['id']}
            })
    return tests

# Question Generation and Follow-up
def generate_followup_question(main_symptom, screening_info, previous_questions, relevant_quote):
    prompt = f"""Based on the symptom: "{main_symptom}" and this information: {relevant_quote['text']},
    generate a specific follow-up question for the patient.
    
    The question should:
    1. Be different from these previous questions: {previous_questions}
    2. Help understand the patient's condition better
    3. Be clear and direct
    
    Format your response EXACTLY like this example:
    QUESTION: How long has your {main_symptom} been present?
    OPTIONS:
    1. Less than an hour
    2. A few hours
    3. Since yesterday
    4. More than a day"""
    
    return groqCall(prompt)

# Advice Generation and Processing
def generate_advice(main_symptom, screening_info, symptom_info, test_results, relevant_quote):
    """Generate patient advice"""
    prompt = f"""Based on this patient's condition:
    Symptom: {main_symptom}
    Screening info: {screening_info}
    Symptoms observed: {symptom_info}
    Test results: {test_results}
    
    Provide 3 specific pieces of advice based on this information: {relevant_quote['text']}
    
    Format your response as follows:
    1. [Specific advice point 1]
    Citation: [Brief summary of supporting evidence]
    
    2. [Specific advice point 2]
    Citation: [Brief summary of supporting evidence]
    
    3. [Specific advice point 3]
    Citation: [Brief summary of supporting evidence]"""
    
    return groqCall(prompt)

def generate_final_advice(main_symptom, screening_info, symptoms, tests, symptom_embedding, relevant_quotes):
    """Generate final advice and recommendations"""
    symptom_info = [f"Symptom: {s['question']} Answer: {s['answer']}" for s in symptoms]
    test_results = []
    for test in tests:
        test_results.append(f"Test: {test['test_name']}")
        for i, (instruction, result) in enumerate(zip(test['instructions'], test['results']), 1):
            test_results.append(f"  Step {i}: {instruction.split('.', 1)[1].strip()} - Result: {result}")
    
    advice_quote = find_most_relevant_quote(symptom_embedding, relevant_quotes)
    advice = generate_advice(main_symptom, screening_info, symptom_info, test_results, advice_quote)
    
    if advice:
        advice_list = advice.split('\n\n')
        return [{
            "advice": item.split('Citation:')[0].strip(),
            "citation_summary": item.split('Citation:')[1].strip() if 'Citation:' in item else "",
            "relevant_quote": {"text": advice_quote['text'], "id": advice_quote['id']}
        } for item in advice_list if item.strip()]
    return []

def parse_llm_response(response_text, main_symptom):
    """Parse LLM response for follow-up questions"""
    try:
        # Split response into parts
        parts = response_text.lower().split('options:')
        if len(parts) != 2:
            raise ValueError("Invalid response format")

        # Extract question
        question_part = parts[0].strip()
        question = question_part.split('question:', 1)[-1].strip()

        # Extract options
        options_text = parts[1].strip()
        options = []
        for line in options_text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                option_text = line.split('.', 1)[-1].strip()
                if option_text:
                    options.append(option_text)

        # Always add custom option
        options.append("Custom...")

        return {
            "question": question.capitalize(),
            "options": [{"id": i+1, "text": opt.capitalize()} for i, opt in enumerate(options)]
        }
    except Exception as e:
        print(f"\n⚠️ Warning: Question parsing issue: {str(e)}")
        return {
            "question": f"How severe is your {main_symptom}?",
            "options": [
                {"id": 1, "text": "Mild"},
                {"id": 2, "text": "Moderate"},
                {"id": 3, "text": "Severe"},
                {"id": 4, "text": "Very Severe"},
                {"id": 5, "text": "Custom..."}
            ]
        }

def gather_followup_symptoms(main_symptom, screening_info, symptom_embedding, relevant_quotes):
    """Gather follow-up symptoms with enhanced question handling"""
    symptoms = []
    total_questions = 3
    print("\n=== Follow-up Assessment ===")
    
    for i in range(total_questions):
        print(f"\nQuestion {i+1}/{total_questions}")
        symptom_quote = find_most_relevant_quote(symptom_embedding, relevant_quotes)
        response = generate_followup_question(
            main_symptom, 
            screening_info, 
            [s['question'] for s in symptoms], 
            symptom_quote
        )
        
        if response:
            question_data = parse_llm_response(response, main_symptom)
            print(f"\n{question_data['question']}")
            print("\nOptions:")
            for opt in question_data['options']:
                print(f"{opt['id']}. {opt['text']}")
            
            while True:
                try:
                    choice = int(input("\nEnter option number: "))
                    if 1 <= choice <= len(question_data['options']):
                        if choice == len(question_data['options']):  # Custom option
                            custom_value = input("\nEnter custom response: ")
                            print(f"\n✓ Recorded: {custom_value}")
                            answer = custom_value
                        else:
                            answer = question_data['options'][choice-1]['text']
                            print(f"\n✓ Selected: {answer}")
                        
                        symptoms.append({
                            "question": question_data['question'],
                            "answer": answer,
                            "relevant_quote": {"text": symptom_quote['text'], "id": symptom_quote['id']}
                        })
                        break
                    else:
                        print("\n❌ Invalid option number.")
                except ValueError:
                    print("\n❌ Please enter a valid number.")
    
    return symptoms

# Utility Functions
def update_json(data, filename="conversation_output.json"):
    """Save conversation data"""
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Conversation data saved to: {filepath}")

def display_welcome():
    """Show welcome message"""
    print("\n" + "="*50)
    print("Welcome to the Community Health Worker Assistant!")
    print("="*50)
    print("\nThis system will help assess patient symptoms through a series of questions.")
    print("For most questions, select from the provided numbered options.")
    print("For symptoms and age, you can enter the information directly.")
    print("A custom option (5) is available for most responses.")
    input("\nPress Enter to begin...")

# Main Function
def main():
    """Main program loop"""
    display_welcome()
    conversation_data = {"conversations": []}
    index = getIndex()
    
    while True:
        print("\n=== New Patient Assessment ===")
        screening_answers = []
        
        # Gather initial screening information
        for question in questions_init:
            print(f"\n{question['question']}")
            answer = get_valid_input(
                question['type'],
                question.get('options'),
                question.get('range')
            )
            screening_answers.append({"question": question['question'], "answer": answer})
        
        main_symptom = next((answer['answer'] for answer in screening_answers 
                           if answer['question'] == "What brings the patient here?"), "")
        
        if main_symptom.upper() == 'STOP':
            print("\n=== Session Ended ===")
            print("Thank you for using the Health Worker Assistant. Goodbye!")
            break
        
        conversation = {
            "screening": screening_answers,
            "followup_symptoms": [],
            "tests": [],
            "advice": [],
            "conversation_followups": []
        }
        
        try:
            print("\n=== Processing Information ===")
            symptom_embedding = get_embedding(main_symptom)
            relevant_quotes = vectorQuotes(symptom_embedding, index)
            screening_info = ", ".join([f"{a['question']}: {a['answer']}" for a in screening_answers])
            
            # Gather additional symptoms
            followup_symptoms = gather_followup_symptoms(main_symptom, screening_info, symptom_embedding, relevant_quotes)
            conversation["followup_symptoms"] = followup_symptoms
            
            # Perform tests
            tests = perform_tests(main_symptom, screening_info, followup_symptoms, symptom_embedding, relevant_quotes)
            conversation["tests"] = tests
            
            # Generate and display advice
            print("\n=== Assessment Results and Recommendations ===")
            advice = generate_final_advice(main_symptom, screening_info, followup_symptoms, tests, symptom_embedding, relevant_quotes)
            conversation["advice"] = advice
            
            for i, item in enumerate(advice, 1):
                print(f"\nRecommendation {i}:")
                print(f"➤ {item['advice']}")
                print(f"   Citation: {item['citation_summary']}")
            
            # Save conversation data
            conversation_data["conversations"].append(conversation)
            update_json(conversation_data)
            
            # Ask about continuing
            print("\n=== Session Complete ===")
            continue_options = [
                {"id": 1, "text": "Start new patient assessment"},
                {"id": 2, "text": "Exit program"}
            ]
            continue_answer = get_valid_input(
                "MC",
                options=continue_options
            )
            
            if continue_answer == "Exit program":
                print("\nThank you for using the Health Worker Assistant. Goodbye!")
                break
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again with a different input.")
            error_options = [
                {"id": 1, "text": "Try again"},
                {"id": 2, "text": "Exit program"}
            ]
            error_answer = get_valid_input(
                "MC",
                options=error_options
            )
            if error_answer == "Exit program":
                print("\nThank you for using the Health Worker Assistant. Goodbye!")
                break

if __name__ == "__main__":
    main()

