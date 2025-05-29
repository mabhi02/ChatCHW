from typing import Dict, List, Any, Optional
import sys
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai

# Load environment variables and initialize clients
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Keep existing questions_init list...
questions_init = [
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
]

# Initialize array to store exam names and procedures
runExamNames = []

def print_options(options: List[Dict[str, Any]]) -> None:
    """Print formatted options for multiple choice questions."""
    for option in options:
        print(f"  {option['id']}: {option['text']}")

def validate_num_input(value: str, range_data: Dict[str, int]) -> Optional[int]:
    """Validate numeric input against given range."""
    try:
        num = int(value)
        if range_data['min'] <= num <= range_data['max']:
            return num
        return None
    except ValueError:
        return None

def validate_mc_input(value: str, options: List[Dict[str, Any]]) -> Optional[str]:
    """Validate multiple choice input against given options."""
    valid_ids = [str(opt['id']) for opt in options]
    return value if value in valid_ids else None

def get_initial_responses() -> List[Dict[str, Any]]:
    """Gather responses for initial screening questions."""
    responses = []
    
    print("\nMedical Assessment Initial Questions")
    print("===================================")
    
    for question in questions_init:
        while True:
            print(f"\n{question['question']}")
            
            if question['type'] in ['MC', 'YN', 'MCM']:
                print_options(question['options'])
                answer = input("Enter your choice (enter the number or id): ").strip()
                
                if question['type'] == 'MCM':
                    print("For multiple selections, separate with commas (e.g., 1,2,3)")
                    if ',' in answer:
                        answers = answer.split(',')
                        valid = all(validate_mc_input(a.strip(), question['options']) for a in answers)
                        if valid:
                            responses.append({
                                "question": question['question'],
                                "answer": [a.strip() for a in answers],
                                "type": question['type']
                            })
                            break
                    else:
                        if validate_mc_input(answer, question['options']):
                            responses.append({
                                "question": question['question'],
                                "answer": [answer],
                                "type": question['type']
                            })
                            break
                else:
                    if validate_mc_input(answer, question['options']):
                        if answer == "5":
                            custom_answer = input("Please specify: ").strip()
                            responses.append({
                                "question": question['question'],
                                "answer": custom_answer,
                                "type": question['type']
                            })
                        else:
                            selected_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                            responses.append({
                                "question": question['question'],
                                "answer": selected_text,
                                "type": question['type']
                            })
                        break
                
            elif question['type'] == 'NUM':
                answer = input(f"Enter a number between {question['range']['min']} and {question['range']['max']}: ")
                if validated_num := validate_num_input(answer, question['range']):
                    responses.append({
                        "question": question['question'],
                        "answer": validated_num,
                        "type": question['type']
                    })
                    break
                    
            elif question['type'] == 'FREE':
                answer = input("Enter your response (type your answer and press Enter): ").strip()
                if answer:
                    responses.append({
                        "question": question['question'],
                        "answer": answer,
                        "type": question['type']
                    })
                    break
            
            print("Invalid input, please try again.")
    
    return responses

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's API."""
    try:
        response = openai.Embedding.create(
            input=text,
            engine="text-embedding-3-small"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def vectorQuotes(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search vector DB and return relevant matches."""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [{"text": match['metadata']['text'], "id": match['id']} for match in results['matches']]
    except Exception as e:
        print(f"Error searching vector DB: {e}")
        return []

def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> bool:
    """Judge if the current question is too similar to previous questions."""
    if not followup_responses:
        return False
        
    try:
        current_embedding = get_embedding(current_question)
        if not current_embedding:
            return False
            
        total_similarity = 0
        for resp in followup_responses:
            prev_question = resp['question']
            prev_embedding = get_embedding(prev_question)
            if prev_embedding:
                similarity = sum(a * b for a, b in zip(current_embedding, prev_embedding))
                magnitude1 = sum(a * a for a in current_embedding) ** 0.5
                magnitude2 = sum(b * b for b in prev_embedding) ** 0.5
                if magnitude1 * magnitude2 != 0:
                    similarity = similarity / (magnitude1 * magnitude2)
                    total_similarity += similarity
                    print(f"Similarity score with '{prev_question}': {similarity:.3f}")
                    
        print(f"Total similarity score: {total_similarity:.3f}")
        return total_similarity > 2.8
        
    except Exception as e:
        print(f"Error in judge function: {e}")
        return False

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> bool:
    """
    Judge if the current examination is too similar to previous ones or if enough exams have been conducted.
    Returns True if should stop asking for exams.
    """
    if not previous_exams:
        return False
        
    try:
        # Get embedding for current exam
        current_embedding = get_embedding(current_exam)
        if not current_embedding:
            return False
            
        # Calculate similarity with all previous exams
        total_similarity = 0
        for exam in previous_exams:
            prev_exam = exam['examination']
            prev_embedding = get_embedding(prev_exam)
            if prev_embedding:
                similarity = sum(a * b for a, b in zip(current_embedding, prev_embedding))
                magnitude1 = sum(a * a for a in current_embedding) ** 0.5
                magnitude2 = sum(b * b for b in prev_embedding) ** 0.5
                if magnitude1 * magnitude2 != 0:
                    similarity = similarity / (magnitude1 * magnitude2)
                    total_similarity += similarity
                    print(f"Similarity score with '{prev_exam}': {similarity:.3f}")
        
        print(f"Total similarity score: {total_similarity:.3f}")
        # More stringent similarity threshold for exams
        return total_similarity > 1.5 or len(previous_exams) >= 5
        
    except Exception as e:
        print(f"Error in judge_exam function: {e}")
        return False

def get_followup_questions(initial_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate and ask follow-up questions based on initial responses."""
    followup_responses = []
    initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
    
    print("\nBased on your responses, I'll ask some follow-up questions.")
    
    index = pc.Index("who-guide-old")
    
    while True:
        try:
            context = f"Initial complaint: {initial_complaint}\n"
            if followup_responses:
                context += "Previous responses:\n"
                for resp in followup_responses:
                    context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
            
            embedding = get_embedding(context)
            relevant_docs = vectorQuotes(embedding, index)
            
            if not relevant_docs:
                print("Error: Could not generate relevant question.")
                continue
                
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
            
            previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
            prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
            
            Previous questions asked:
            {previous_questions if followup_responses else "No previous questions yet"}
            
            Generate ONE focused, relevant follow-up question that is different from the previous questions.
            Follow standard medical assessment order:
            1. Duration and onset
            2. Characteristics and severity
            3. Associated symptoms
            4. Impact on daily life
            
            Return only the question text.'''
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3
            )
            
            question = completion.choices[0].message.content.strip()
            
            if judge(followup_responses, question):
                print("\nSufficient information gathered. Moving to next phase...")
                break
            
            options_prompt = f'''Generate 4 possible answers for: "{question}"
            Requirements:
            - Clear, concise options
            - Mutually exclusive
            - Cover likely scenarios
            - Include severity levels if applicable
            Return each option on a new line starting with a number (1-4).'''
            
            options_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": options_prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.2
            )
            
            options = []
            for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
                if opt.strip():
                    text = opt.strip()
                    if text[0].isdigit() and text[1] in ['.','-',')']:
                        text = text[2:].strip()
                    options.append({"id": i+1, "text": text})
            
            options.append({"id": 5, "text": "Other (please specify)"})
            
            print(f"\n{question}")
            print_options(options)
            
            while True:
                answer = input("Enter your choice (enter the number): ").strip()
                
                if validate_mc_input(answer, options):
                    if answer == "5":
                        custom_answer = input("Please specify your answer: ").strip()
                        followup_responses.append({
                            "question": question,
                            "answer": custom_answer,
                            "type": "MC"
                        })
                    else:
                        selected_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                        followup_responses.append({
                            "question": question,
                            "answer": selected_text,
                            "type": "MC"
                        })
                    break
                print("Invalid input, please try again.")
                
        except Exception as e:
            print(f"Error generating follow-up question: {e}")
            continue
            
    return followup_responses

def get_followup_exams(initial_responses: List[Dict[str, Any]], followup_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate and recommend basic medical examinations based on responses."""
    exam_responses = []
    initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
    
    print("\nBased on your responses, I'll recommend some basic examinations.")
    
    index = pc.Index("who-guide-old")
    
    while True:
        try:
            # Build context including initial complaint and all previous responses
            context = f"Initial complaint: {initial_complaint}\n"
            context += "Previous responses:\n"
            for resp in followup_responses:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
            
            if exam_responses:
                context += "Previous examinations:\n"
                for exam in exam_responses:
                    context += f"Exam: {exam['examination']}\nResult: {exam['result']}\n"
            
            # Add context about previously run exams
            if runExamNames:
                context += "\nPreviously conducted exams:\n"
                for exam in runExamNames:
                    context += f"Exam: {exam['name']}\nProcedure: {exam['procedure']}\n"
            
            embedding = get_embedding(context)
            relevant_docs = vectorQuotes(embedding, index)
            
            if not relevant_docs:
                print("Error: Could not generate relevant examination.")
                continue
            
            combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
            
            previous_exams = "\n".join([f"- {exam['examination']}" for exam in exam_responses])
            prompt = f'''Based on the patient's initial complaint: "{initial_complaint}" and previous examinations:
            {previous_exams if exam_responses else "No examinations yet"}
            
            Already conducted exams:
            {str(runExamNames) if runExamNames else "No exams conducted yet"}
            
            Consider this is a resource-constrained setting in a developing country.
            Recommend ONE basic examination that:
            1. Requires minimal equipment
            2. Can be performed in a basic clinic
            3. Is essential for diagnosis
            4. Does not require advanced technology
            5. Has not been conducted yet (check already conducted exams)
            
            Return in this exact format:
            Examination: [name]
            Procedure: [detailed step by step procedure]'''
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3
            )
            
            examination = completion.choices[0].message.content.strip()
            
            # Get similarity score
            should_end = judge_exam(exam_responses, examination)
            
            
            # Parse examination name and procedure
            try:
                exam_lines = examination.split('\n')
                exam_name = exam_lines[0].split('Examination:')[1].strip()
                exam_procedure = '\n'.join(exam_lines[1:]).split('Procedure:')[1].strip()
                
                # Store in runExamNames
                runExamNames.append({
                    "name": exam_name,
                    "procedure": exam_procedure
                })
            except:
                print("Error parsing examination format")
                continue
            
            # Generate possible findings/results
            results_prompt = f'''Generate 4 possible findings for the examination: "{exam_name}"
            Requirements:
            - Include normal finding
            - Include common abnormal findings
            - Clear, observable results
            - Avoid technical jargon
            Return each finding on a new line starting with a number (1-4).'''
            
            results_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": results_prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.2
            )
            
            options = []
            for i, opt in enumerate(results_completion.choices[0].message.content.strip().split('\n')):
                if opt.strip():
                    text = opt.strip()
                    if text[0].isdigit() and text[1] in ['.','-',')']:
                        text = text[2:].strip()
                    options.append({"id": i+1, "text": text})
            
            options.append({"id": 5, "text": "Other (please specify)"})
            
            print(f"\nRecommended Examination:\nExamination: {exam_name}")
            print(f"Procedure: {exam_procedure}")
            print("\nPlease select the finding/result:")
            print_options(options)
            
            while True:
                answer = input("Enter the finding (enter the number): ").strip()
                
                if validate_mc_input(answer, options):
                    if answer == "5":
                        custom_result = input("Please specify the finding: ").strip()
                        exam_responses.append({
                            "examination": examination,
                            "result": custom_result,
                            "type": "EXAM"
                        })
                    else:
                        selected_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                        exam_responses.append({
                            "examination": examination,
                            "result": selected_text,
                            "type": "EXAM"
                        })
                    # If this was the last exam (similarity threshold reached), break the outer loop
                    if should_end:
                        print("\nSufficient examinations completed. Moving to next phase...")
                        return exam_responses
                    break
                print("Invalid input, please try again.")
                
        except Exception as e:
            print(f"Error generating examination: {e}")
            continue
            
    return exam_responses

def main():
    try:
        # Get initial responses
        initial_responses = get_initial_responses()
        print("\nThank you for providing your information. Here's what we recorded:\n")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")
        
        # Get follow-up responses
        followup_responses = get_followup_questions(initial_responses)
        
        print("\nFollow-up responses recorded:\n")
        for resp in followup_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")
        
        # Get examination responses
        exam_responses = get_followup_exams(initial_responses, followup_responses)
        
        print("\nExamination findings recorded:\n")
        for exam in exam_responses:
            print(f"Examination: {exam['examination']}")
            print(f"Finding: {exam['result']}\n")

        # Prepare context for diagnosis
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        # Get relevant documents for diagnosis
        embedding = get_embedding(initial_complaint)
        relevant_docs = vectorQuotes(embedding, pc.Index("who-guide-old"))
        context_chunks = " ".join([doc["text"] for doc in relevant_docs[:3]])

        # Create diagnosis prompt
        diagnosis_context = {
            "complaint": initial_complaint,
            "responses": {
                "initial": initial_responses,
                "followup": followup_responses,
                "exams": exam_responses
            }
        }

        diagnosis_prompt = f'''Based on the following patient information, provide a clear and specific diagnosis.

Patient's initial complaint: "{initial_complaint}"

Follow-up responses:
{', '.join([f"Q: {resp['question']} A: {resp['answer']}" for resp in followup_responses])}

Examination findings:
{', '.join([f"Exam: {exam['examination']} Finding: {exam['result']}" for exam in exam_responses])}

Additional context:
{context_chunks}

Return ONLY the most likely diagnosis in a clear, concise manner. No explanations or additional information.'''

        # Get diagnosis
        diagnosis_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": diagnosis_prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.2
        )
        
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        print("\nDiagnosis:")
        print("==========")
        print(diagnosis)

        # Get relevant documents for treatment based on diagnosis
        diagnosis_embedding = get_embedding(diagnosis)
        treatment_docs = vectorQuotes(diagnosis_embedding, pc.Index("who-guide-old"))
        treatment_chunks = " ".join([doc["text"] for doc in treatment_docs[:3]])

        # Create treatment prompt
        treatment_prompt = f'''Based on the diagnosis and patient information, recommend appropriate treatments.

Diagnosis: {diagnosis}

Patient context:
- Initial complaint: {initial_complaint}
- Age: {next((resp['answer'] for resp in initial_responses if resp['question'] == "What is the patient's age?"), "Unknown")}
- Findings: {', '.join([f"{exam['examination']}: {exam['result']}" for exam in exam_responses])}

Relevant medical context:
{treatment_chunks}

Provide a clear treatment plan considering this is a resource-constrained setting.
Focus on:
1. Immediate interventions
2. Medications (if needed)
3. Home care instructions
4. Follow-up recommendations'''

        # Get treatment plan
        treatment_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": treatment_prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.3
        )

        treatment = treatment_completion.choices[0].message.content.strip()
        print("\nRecommended Treatment Plan:")
        print("=========================")
        print(treatment)
        
        # Return all responses including diagnosis and treatment
        return {
            "initial_responses": initial_responses,
            "followup_responses": followup_responses,
            "examination_responses": exam_responses,
            "diagnosis": diagnosis,
            "treatment": treatment
        }
            
    except KeyboardInterrupt:
        print("\nAssessment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()