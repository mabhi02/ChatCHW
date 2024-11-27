import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
import ast

# Initialize environment and API clients
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize global variables
used_citations = set()  # Global set to store used citation IDs
citation_data = pd.read_csv("chunks_with_pages.csv")

def handle_screening_questions():
    questions = [
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
            "range": {
                "min": 0,
                "max": 120,
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
            "type": "FREE"
        }
    ]

    screening_data = []

    for q in questions:
        print(f"\n{q['question']}")
        
        if q['type'] == "MC":
            print("Options:")
            for opt in q['options']:
                print(f"{opt['id']}. {opt['text']}")
            
            while True:
                try:
                    choice = int(input("Enter option number: "))
                    if 1 <= choice <= len(q['options']):
                        answer = q['options'][choice-1]['text']
                        print(f"✓ Selected: {answer}")
                        screening_data.append({"question": q['question'], "answer": answer})
                        break
                    print("❌ Invalid option number.")
                except ValueError:
                    print("❌ Please enter a valid number.")

        elif q['type'] == "NUM":
            while True:
                try:
                    age = int(input(f"Enter a number between {q['range']['min']} and {q['range']['max']} {q['range']['unit']}: "))
                    if q['range']['min'] <= age <= q['range']['max']:
                        print(f"✓ Recorded: {age} {q['range']['unit']}")
                        screening_data.append({"question": q['question'], "answer": str(age)})
                        break
                    print(f"❌ Please enter a number between {q['range']['min']} and {q['range']['max']}")
                except ValueError:
                    print("❌ Please enter a valid number.")

        elif q['type'] == "MCM":
            print("Options:")
            for opt in q['options']:
                print(f"{opt['id']}. {opt['text']}")
            
            while True:
                try:
                    choices = input("Enter option numbers, separated by commas: ")
                    chosen_ids = [int(x.strip()) for x in choices.split(',')]
                    valid_choices = [opt for opt in q['options'] if opt['id'] in chosen_ids]
                    
                    if len(valid_choices) == len(chosen_ids):
                        answer = ", ".join(opt['text'] for opt in valid_choices)
                        print(f"✓ Selected: {answer}")
                        screening_data.append({"question": q['question'], "answer": answer})
                        break
                    print("❌ Invalid option(s) selected.")
                except ValueError:
                    print("❌ Please enter valid numbers separated by commas.")

        elif q['type'] == "FREE":
            answer = input("Please describe the patient's symptoms: ")
            if answer.strip():
                print(f"✓ Recorded: {answer}")
                screening_data.append({"question": q['question'], "answer": answer})
            else:
                print("❌ Please provide a description.")

    return screening_data


def enhance_query_for_similarity(query, screening_data):
    symptoms = next((item['answer'] for item in screening_data if item['question'] == "What brings the patient here?"), "")
    age = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's age?"), "")
    sex = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's sex?"), "")

    prompt = f"""You are assisting in a medical chatbot that focuses on local remedies and treatments not Western medicinial solutions or names. Based on the patient query: "{symptoms}", rewrite the query to make it more detailed and specific, ensuring it focuses on symptoms and conditions relevant to the local medical guides. Avoid ambiguous terms or interpretations, and ensure the query is framed to match information in the database that addresses the described symptom. The goal is to find the most relevant remedies and treatments related to the symptom without veering into unrelated topics."""

    #altPrompt = """Answer the query: """ + query + " as if you a doctor in a third world country with only acess to basic treatement methods (pulse, temperature, etc). Do not ask about causes, followup questions, or be cordial just list off possible solutions in relation to the problem"

    enhanced_query = groqCall(prompt)
    if not enhanced_query:
        return f"{symptoms}, {sex}, age {age}"
        
    return enhanced_query

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

def getIndex():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("final-asha")
    return index

def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, engine="text-embedding-3-small")
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        return None

def perform_similarity_search(screening_data, index):
    # Get original query and enhanced version
    original_query = next((item['answer'] for item in screening_data 
                         if item['question'] == "What brings the patient here?"), "")
    enhanced_query = enhance_query_for_similarity(original_query, screening_data)
    #enhanced_query = original_query

    #print("Enhanced query is ", enhanced_query)
    
    try:
        # Get embedding for enhanced query
        embedding = get_embedding(enhanced_query)
        if not embedding:
            return []
            
        # Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Process results
        quotes = []
        for match in results['matches']:
            used_citations.add(match['id'])
            quotes.append({
                "text": match['metadata']['text'],
                "id": match['id'],
                "score": match['score']
            })
            
        
        #print(f"Found {len(quotes)} relevant quotes")
        #for i, quote in enumerate(quotes, 1):
            #print(f"\nMatch {i} (score: {quote['score']:.3f}):")
            #print(f"➤ {quote['text'][:150]}...")
            
        return quotes
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []

def get_page_citations(val):
    # Initialize an empty list to store pages
    val = [int(v) for v in val]
    df = pd.read_csv('chunks_with_pages.csv')
    pages = []
    
    # Iterate through each id in val
    for id_val in val:
        # Access the page_numbers array for the corresponding id
        page_numbers = df.loc[df['id'] == id_val, 'page_numbers'].values
        if len(page_numbers) > 0:
            # Safely evaluate the string as a Python list, if necessary
            try:
                page_list = ast.literal_eval(page_numbers[0]) if isinstance(page_numbers[0], str) else page_numbers[0]
                # Extend the pages list with the elements of page_list
                pages.extend(page_list)
            except (ValueError, SyntaxError):
                print(f"Skipping invalid page_numbers format for id {id_val}: {page_numbers[0]}")
    
    # Get unique values and sort them
    unique_sorted_pages = sorted(set(pages))
    
    return unique_sorted_pages

def generate_diagnostic_tests(query, screening_data):
    symptoms = next((item['answer'] for item in screening_data if item['question'] == "What brings the patient here?"), "")
    age = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's age?"), "")
    sex = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's sex?"), "")

    prompt = f"""For a {age}-year-old {sex} patient with symptoms: "{symptoms}"
Create 2 simple diagnostic tests. Each test should have exactly 3 measurable parameters.
Each measurement should be simple and straightforward.

Format exactly as:
Test 1: [name]
Description: [1-line description]
Measurements:
1. [simple measurement parameter] (MEASURE)
2. [simple measurement parameter] (MEASURE)
3. [simple measurement parameter] (MEASURE)

Test 2: [name]
Description: [1-line description]
Measurements:
1. [simple measurement parameter] (MEASURE)
2. [simple measurement parameter] (MEASURE)
3. [simple measurement parameter] (MEASURE)"""

    try:
        return groqCall(prompt)
    except Exception as e:
        print(f"❌ Error generating tests: {str(e)}")
        return None

def generate_step_options(step_text):
    """Generate simple, single-parameter options for each measurement"""
    prompt = f"""For this measurement: "{step_text}"
Generate 3 clear, simple options that cover the range of possible findings.
Each option should be a single, specific finding.

Format exactly as:
1. [simple finding]
2. [simple finding]
3. [simple finding]

Example format:
1. Normal temperature (36.5-37.5°C)
2. Mild fever (37.6-38.3°C)
3. High fever (>38.3°C)"""

    try:
        return groqCall(prompt)
    except Exception as e:
        print(f"❌ Error generating options: {str(e)}")
        return None

def perform_test_steps(test_name, steps):
    """Handle user input for each test step and store results"""
    test_record = [f"\n=== {test_name} Results ===\n"]
    
    for step in steps:
        print(f"\nAssessing: {step}")
        
        options = generate_step_options(step)
        if not options:
            continue
            
        # Display options
        print("\nOptions:")
        print(options)
        print("4. Custom...")
        
        # Get user input
        while True:
            try:
                choice = int(input("Enter option number: "))
                if 1 <= choice <= 4:
                    if choice == 4:
                        result = input("Enter custom value: ")
                    else:
                        # Extract just the option text without the number
                        options_list = [line.strip() for line in options.split('\n') if line.strip() and line[0].isdigit()]
                        result = options_list[choice-1].split('. ', 1)[1]
                    print(f"✓ Recorded: {result}")
                    test_record.append(f"Step: {step}\nResult: {result}\n")
                    break
                else:
                    print("❌ Invalid option number.")
            except ValueError:
                print("❌ Please enter a valid number.")
            except IndexError:
                print("❌ Error processing option. Please try again.")
                
    return "\n".join(test_record)

def parse_test_steps(test_output):
    """Parse the Groq test output into structured test data"""
    if not test_output:
        return []
        
    tests = []
    current_test = {}
    current_steps = []
    
    for line in test_output.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Test'):
            if current_test:
                current_test['steps'] = current_steps
                tests.append(current_test)
                current_steps = []
            current_test = {'name': line}
        elif line.startswith('Description:'):
            current_test['description'] = line.replace('Description:', '').strip()
        elif line.startswith('Steps:'):
            continue
        elif line[0].isdigit() and '. ' in line:
            step = line.split('. ')[1].strip()
            current_steps.append(step)
    
    # Add the last test
    if current_test:
        current_test['steps'] = current_steps
        tests.append(current_test)
        
    return tests

def generate_solutions(screening_data, quotes, test_results):
    """Generate treatment solutions based on all collected data"""
    symptoms = next((item['answer'] for item in screening_data if item['question'] == "What brings the patient here?"), "")
    age = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's age?"), "")
    sex = next((item['answer'] for item in screening_data if item['question'] == "What is the patient's sex?"), "")
    
    quote_text = "\n".join([f"Reference {i+1}: {quote['text']}" for i, quote in enumerate(quotes)])
    
    prompt = f"""Based on the following patient information and test results, provide 3 possible solutions/treatments in order of recommendation:

Patient Information:
- Age: {age}
- Sex: {sex}
- Primary Symptoms: {symptoms}

Relevant Medical References:
{quote_text}

Test Results:
{test_results}

Provide 3 solutions in order of recommendation. For each solution:
1. State the likely diagnosis
2. Recommend specific treatment steps
3. Provide warning signs to watch for
4. Suggest when to seek additional medical care

Format as:
SOLUTION 1 (Most Recommended):
Diagnosis: [diagnosis]
Treatment:
- [step 1]
- [step 2]
Warning Signs:
- [sign 1]
- [sign 2]
Seek Care If: [conditions]

[Repeat for Solutions 2 and 3]"""

    try:
        return groqCall(prompt)
    except Exception as e:
        print(f"❌ Error generating solutions: {str(e)}")
        return None

def main():
    try:
        print("\n=== Health Worker Assessment System ===")
        index = getIndex()
        
        screening_data = handle_screening_questions()
        main_symptom = next((item['answer'] for item in screening_data 
                           if item['question'] == "What brings the patient here?"), "")
        
        if main_symptom.upper() == 'STOP':
            return
            
        print("\n=== Processing Query ===")
        quotes = perform_similarity_search(screening_data, index)
        
        citationPages = []
        if quotes:
            print("\n=== Relevant References ===")
            for i, quote in enumerate(quotes, 1):
                citationPages.append(quote['id'])
                
            # Generate and parse diagnostic tests
            print("\n=== Diagnostic Tests ===")
            tests_output = generate_diagnostic_tests(main_symptom, screening_data)
            if tests_output:
                #print("\nGenerated Tests:")
                #print(tests_output)
                
                # Parse and run the tests
                parsed_tests = parse_test_steps(tests_output)
                all_test_logs = []
                
                print("\n=== Beginning Test Procedures ===")
                for test in parsed_tests:
                    test_logs = perform_test_steps(test['name'], test['steps'])
                    all_test_logs.append(test_logs)
                
                print("\n=== Complete Test Results ===")
                test_results = "\n".join(all_test_logs)
                print(test_results)
                
                # Generate solutions based on all collected data
                print("\n=== Recommended Solutions ===")
                solutions = generate_solutions(screening_data, quotes, test_results)
                if solutions:
                    print(solutions)

            # Get unique pages from all citations
            pages_set = get_page_citations(citationPages)
            if pages_set:
                print("\n=== Citation Summary ===")
                print("Citations used:", pages_set)
            else:
                print("\nNo page citations found.")
            
        else:
            print("No relevant references found.")
            
    except Exception as e:
        print(f"❌ Error in main program: {str(e)}")

if __name__ == "__main__":
    main()