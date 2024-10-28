import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def update_json(data, filename="conversation_output.json"):
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON updated: {filepath}")

def ask_screening_questions():
    screening_answers = []
    for question in questions_init:
        print(f"\nCommunity Health Worker: {question['question']}")
        if question['type'] in ['MC', 'YN', 'MCM']:
            for option in question['options']:
                print(f"{option['id']}. {option['text']}")
            answer = input("Enter the number(s) of your choice(s), separated by commas if multiple: ")
        elif question['type'] == 'NUM':
            answer = input(f"Enter a number between {question['range']['min']} and {question['range']['max']} {question['range']['unit']}: ")
        else:  # FREE
            answer = input("Enter your response: ")
        screening_answers.append({"question": question['question'], "answer": answer})
    return screening_answers

def gather_followup_symptoms(main_symptom, screening_info, symptom_embedding, relevant_quotes):
    symptoms = []
    for _ in range(3):  # Ask for 3 follow-up questions
        symptom_quote = find_most_relevant_quote(symptom_embedding, relevant_quotes)
        question = generate_followup_question(main_symptom, screening_info, [s['question'] for s in symptoms], symptom_quote)
        if question:
            print(f"Community Health Worker: {question}")
            answer = input("Patient: ")
            symptoms.append({
                "question": question,
                "answer": answer,
                "relevant_quote": {"text": symptom_quote['text'], "id": symptom_quote['id']}
            })
    return symptoms

def perform_tests(main_symptom, screening_info, symptoms, symptom_embedding, relevant_quotes):
    tests = []
    symptom_info = [f"Symptom: {s['question']} Answer: {s['answer']}" for s in symptoms]
    for _ in range(2):  # Perform 2 tests
        test_quote = find_most_relevant_quote(symptom_embedding, relevant_quotes)
        previous_tests = [t['test_name'] for t in tests]
        test_info = generate_test(main_symptom, screening_info, symptom_info, test_quote, previous_tests)
        if test_info:
            print(f"\nCommunity Health Worker: Let's perform this test:")
            test_lines = test_info.split('\n')
            test_name = test_lines[0].replace('Test: ', '').strip()
            test_description = test_lines[1].replace('Description: ', '').strip()
            print(f"Test: {test_name}")
            print(f"Description: {test_description}")
            print("Instructions:")
            
            instructions = []
            results = []
            for line in test_lines[3:]:  # Skip the "Instructions:" line
                if line.strip() and line[0].isdigit():
                    print(line)
                    instructions.append(line)
                    result = input("Enter result for this step: ")
                    results.append(result)
            
            tests.append({
                "test_name": test_name,
                "test_description": test_description,
                "instructions": instructions,
                "results": results,
                "relevant_quote": {"text": test_quote['text'], "id": test_quote['id']}
            })
    return tests

def generate_final_advice(main_symptom, screening_info, symptoms, tests, symptom_embedding, relevant_quotes):
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
            "advice": item.split('Citation summary:')[0].strip(),
            "citation_summary": item.split('Citation summary:')[1].strip() if 'Citation summary:' in item else "",
            "relevant_quote": {"text": advice_quote['text'], "id": advice_quote['id']}
        } for item in advice_list if item.strip()]
    return []

def main():
    print("Welcome to the Community Health Worker Assistant!")
    
    conversation_data = {"conversations": []}
    index = getIndex()
    
    while True:
        print("\nLet's start with some screening questions.")
        screening_answers = ask_screening_questions()
        
        main_symptom = next((answer['answer'] for answer in screening_answers if answer['question'] == "What brings the patient here?"), "")
        
        if main_symptom.upper() == 'STOP':
            print("Community Health Worker: Thank you for using the Assistant. Take care!")
            break
        
        conversation = {
            "screening": screening_answers,
            "followup_symptoms": [],
            "tests": [],
            "advice": [],
            "conversation_followups": []
        }
        
        try:
            symptom_embedding = get_embedding(main_symptom)
            relevant_quotes = vectorQuotes(symptom_embedding, index)
            
            screening_info = ", ".join([f"{a['question']}: {a['answer']}" for a in screening_answers])
            
            # Ask follow-up questions
            print("\nCommunity Health Worker: Let's gather some more information.")
            followup_symptoms = gather_followup_symptoms(main_symptom, screening_info, symptom_embedding, relevant_quotes)
            conversation["followup_symptoms"] = followup_symptoms
            
            # Perform tests
            print("\nCommunity Health Worker: Now, let's perform some tests.")
            tests = perform_tests(main_symptom, screening_info, followup_symptoms, symptom_embedding, relevant_quotes)
            conversation["tests"] = tests
            
            # Generate advice
            print("\nCommunity Health Worker: Based on the information provided, here's my advice:")
            advice = generate_final_advice(main_symptom, screening_info, followup_symptoms, tests, symptom_embedding, relevant_quotes)
            conversation["advice"] = advice
            
            for i, item in enumerate(advice, 1):
                print(f"{i}. {item['advice']}")
                print(f"   Citation summary: {item['citation_summary']}")
            
            # Generate and ask follow-up questions
            conversation_summary = f"Main symptom: {main_symptom}, Screening info: {screening_info}, " \
                                   f"Followup symptoms: {followup_symptoms}, Tests: {tests}, Advice: {advice}"
            
            for _ in range(2):  # Ask 2 follow-up questions
                followup = generate_conversation_followup(conversation_summary)
                print(f"\nCommunity Health Worker: {followup}")
                response = input("Patient: ")
                conversation["conversation_followups"].append({"followup": followup, "response": response})
                conversation_summary += f", Followup: {followup}, Response: {response}"
            
            conversation_data["conversations"].append(conversation)
            update_json(conversation_data)
            
            print("\nIs there anything else I can help you with?")
        
        except Exception as e:
            print(f"An error occurred while processing your request: {str(e)}")
            print("Please try again with a different symptom or more detailed information.")

if __name__ == "__main__":
    main()