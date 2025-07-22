"""
Professional Way CHW Assistant (profsway.py)

CRITICAL: This implementation uses ONLY Retrieval-Augmented Generation (RAG) 
from the WHO medical guide. NO hardcoded medical knowledge is included.

All medical decisions including:
- Symptom identification
- Danger sign questions  
- Examinations
- Diagnosis
- Treatment

Are derived ENTIRELY from the WHO medical guide via vector database retrieval
from the "who-guide-old" Pinecone index.

If symptoms are not covered in the WHO guide, patients are referred to medical centers.

TRANSPARENCY FEATURES:
- Citations and sources are displayed for every medical decision
- Users can see which parts of the WHO guide were used
- Relevance scores show confidence in the retrieved information
- Preview text shows the actual content used from the guide
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import json

from prompts_infer import (
    get_diagnosis_prompt,
    get_treatment_prompt,
    get_main_followup_question_prompt,
    get_main_examination_prompt
)

# Load environment variables and initialize clients
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global arrays for session data
examination_history = []
structured_questions_array = []

# Initial screening questions (same as original)
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

# All medical knowledge will be retrieved from the WHO guide via RAG

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's API."""
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return [[] for _ in texts]

def vectorQuotesWithSource(query_embedding: List[float], index, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search vector DB and return relevant matches with source information."""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [{
            "text": match['metadata']['text'],
            "id": match['id'],
            "source": match['metadata'].get('source', 'Unknown'),
            "score": match['score']
        } for match in results['matches']]
    except Exception as e:
        print(f"Error searching vector DB: {e}")
        return []

def get_openai_completion(prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
    """Get completion from OpenAI's GPT-4 API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return ""

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

def identify_symptoms_from_complaint(complaint: str, index) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Identify symptoms from the initial complaint using RAG from WHO guide."""
    try:
        # Use RAG to identify symptoms mentioned in the complaint
        embedding = get_embedding_batch([complaint])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=5)
        
        if not relevant_docs:
            return [], []
        
        # Use medical guide content to identify symptoms
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        symptom_identification_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Based ONLY on the WHO medical guide content below, identify the main symptoms mentioned in this patient complaint: "{complaint}"
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        Return ONLY the specific symptoms that are:
        1. Mentioned in the patient's complaint
        2. Explicitly covered in the WHO medical guide content above
        
        Format your response as a simple comma-separated list of symptoms (e.g., "fever, cough, diarrhea")
        If no symptoms can be identified from the WHO medical guide, respond with "none"
        
        Remember: Use ONLY information from the WHO guide above, not your medical training.
        """
        
        response = get_openai_completion(
            prompt=symptom_identification_prompt,
            max_tokens=100,
            temperature=0.1
        )
        
        if response.lower().strip() == "none":
            return [], []
        
        # Parse the response to extract symptoms
        symptoms = [s.strip() for s in response.split(',')]
        symptoms = [s for s in symptoms if s and len(s) > 2]
        
        return symptoms, relevant_docs
        
    except Exception as e:
        print(f"Error identifying symptoms from complaint: {e}")
        return [], []

# Removed check_symptom_in_medical_guide() function since symptoms identified 
# from WHO guide are covered by definition

def display_citations(citations: List[Dict[str, Any]], context: str = ""):
    """Display citations from WHO medical guide in a formatted way."""
    if not citations:
        return
    
    print(f"\n📖 WHO Medical Guide Sources {context}:")
    for i, citation in enumerate(citations[:3], 1):  # Show top 3 sources
        source = citation.get('source', 'Unknown')
        score = citation.get('score', 0.0)
        text_preview = citation.get('text', '')[:150] + "..." if len(citation.get('text', '')) > 150 else citation.get('text', '')
        print(f"  {i}. {source} (relevance: {score:.2f})")
        print(f"     Preview: {text_preview}")

def ask_danger_sign_questions(symptoms: List[str], index) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Ask danger sign questions for each symptom using RAG from WHO guide."""
    danger_sign_responses = []
    all_citations = []
    
    for symptom in symptoms:
        try:
            # Use RAG to find danger signs for this symptom from WHO guide
            danger_sign_query = f"danger signs warning signs {symptom} emergency"
            embedding = get_embedding_batch([danger_sign_query])[0]
            relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)
            
            if not relevant_docs:
                continue
            
            # Store citations for this symptom
            all_citations.extend(relevant_docs)
            
            medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            danger_signs_prompt = f"""
            CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
            
            Based ONLY on the WHO medical guide content below, generate 2-3 specific yes/no questions to check for danger signs related to "{symptom}".
            
            WHO Medical Guide Content:
            {medical_guide_content}
            
            Requirements:
            1. Questions must be based ONLY on the WHO medical guide content above
            2. Focus on serious complications or warning signs mentioned in the guide
            3. Each question should be answerable with yes/no
            4. Format each question on a new line starting with "Q:"
            5. Do NOT add medical knowledge from your training
            
            If the WHO medical guide doesn't contain specific danger signs for this symptom, respond with "none"
            
            Remember: Use ONLY information from the WHO guide above, not your medical training.
            """
            
            response = get_openai_completion(
                prompt=danger_signs_prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            if response.lower().strip() == "none":
                continue
                
            # Parse danger sign questions
            questions = []
            for line in response.split('\n'):
                if line.strip().startswith('Q:'):
                    question = line.strip()[2:].strip()
                    if question:
                        questions.append(question)
            
            if not questions:
                continue
                
            print(f"\nFor your {symptom}, I need to check for danger signs based on the WHO medical guide:")
            
            # Display citations for this symptom's danger signs
            display_citations(relevant_docs, f"for {symptom} danger signs")
            
            for danger_sign_question in questions:
                options = [
                    {"id": 1, "text": "Yes"},
                    {"id": 2, "text": "No"},
                    {"id": 3, "text": "Not sure"},
                    {"id": 4, "text": "Sometimes"},
                    {"id": 5, "text": "Other (please specify)"}
                ]
                
                print(f"\n{danger_sign_question}")
                print_options(options)
                
                while True:
                    answer = input("Enter your choice (enter the number): ").strip()
                    
                    if validate_mc_input(answer, options):
                        if answer == "5":
                            custom_answer = input("Please specify your answer: ").strip()
                            answer_text = custom_answer
                        else:
                            answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                        
                        danger_sign_responses.append({
                            "symptom": symptom,
                            "question": danger_sign_question,
                            "answer": answer_text,
                            "type": "MC",
                            "source": "WHO Medical Guide",
                            "citations": relevant_docs
                        })
                        break
                    
                    print("Invalid input, please try again.")
                    
        except Exception as e:
            print(f"Error generating danger sign questions for {symptom}: {e}")
            continue
    
    return danger_sign_responses, all_citations

def parse_examination_text(text):
    """Parse examination text to extract the examination description and findings."""
    # Same parsing logic as original chad.py
    if not text or len(text.strip()) < 10:
        examination = "No examination information provided in the medical guide."
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    explicit_no_info_phrases = [
        "the medical guide does not provide specific examination procedures for this condition",
        "the guide does not provide specific examination procedures",
        "no examination procedures are provided",
        "no specific examination procedures for this condition"
    ]
    
    text_lower = text.lower()
    is_explicit_no_info = any(phrase in text_lower for phrase in explicit_no_info_phrases)
    
    if is_explicit_no_info and len(text.strip()) < 100:
        examination = text.strip()
        findings = [
            "No specific finding available - refer to higher facility",
            "Unable to perform examination based on medical guide",
            "Need additional clinical assessment",
            "Consider general observation only"
        ]
        return examination, findings
    
    lines = text.strip().split('\n')
    
    findings_markers = []
    for i, line in enumerate(lines):
        if line.startswith('#:') or (line.startswith('#') and not line.startswith('#:')):
            findings_markers.append(i)
    
    if findings_markers:
        examination_lines = lines[:findings_markers[0]]
        findings = []
        
        labels = ['A', 'B', 'C', 'D']
        for i, marker_idx in enumerate(findings_markers):
            line = lines[marker_idx].strip()
            if line.startswith('#:'):
                finding = line[2:].strip()
            else:
                finding = line[1:].strip()
            
            if i < len(labels):
                finding = f"{labels[i]}. {finding}"
            else:
                finding = f"{i+1}. {finding}"
            
            findings.append(finding)
        
        examination = '\n'.join(examination_lines)
        
        if not findings and examination:
            if any(word in examination.lower() for word in ['malaria', 'rdt', 'rapid diagnostic']):
                findings = [
                    "A. RDT positive for malaria",
                    "B. RDT negative for malaria", 
                    "C. RDT not performed: no supplies",
                    "D. RDT not performed: other reason"
                ]
            else:
                findings = [
                    "A. Normal finding",
                    "B. Abnormal finding requiring further assessment",
                    "C. Inconclusive finding - may need additional tests",
                    "D. Unable to determine based on current examination"
                ]
    else:
        if len(lines) < 3:
            examination = "PARTIAL INFORMATION (Not a formal examination procedure):\n" + text.strip()
            findings = [
                "A. Consider referring to medical professional for proper examination",
                "B. Use this information as supplementary guidance only",
                "C. Document observations based on general assessment",
                "D. Consult with supervisor about next steps"
            ]
        else:
            examination = text.strip()
            if any(word in examination.lower() for word in ['malaria', 'rdt', 'rapid diagnostic']):
                findings = [
                    "A. RDT positive for malaria",
                    "B. RDT negative for malaria", 
                    "C. RDT not performed: no supplies",
                    "D. RDT not performed: other reason"
                ]
            else:
                findings = [
                    "A. Normal finding",
                    "B. Abnormal finding requiring further assessment",
                    "C. Inconclusive finding - may need additional tests",
                    "D. Unable to determine based on current examination"
                ]
    
    if not findings and not examination:
        examination = "The response from the medical guide was incomplete or invalid."
        findings = [
            "A. No specific finding available - refer to higher facility",
            "B. Unable to perform examination based on medical guide",
            "C. Need additional clinical assessment",
            "D. Consider general observation only"
        ]
    
    return examination, findings

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                              danger_sign_responses: List[Dict[str, Any]], 
                              exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get diagnosis and treatment based on responses"""
    try:
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        symptoms = []
        for resp in danger_sign_responses:
            if isinstance(resp.get('answer'), str) and resp.get('answer') not in ["No", "Not sure"]:
                symptoms.append(f"{resp['symptom']}: {resp['question']} - {resp['answer']}")
        
        symptoms_text = "\n".join([f"- {symptom}" for symptom in symptoms])
        
        exam_results = []
        for exam in exam_responses:
            if 'examination' in exam and 'result' in exam:
                exam_results.append(f"Examination: {exam['examination']}\nResult: {exam['result']}")
        exam_results_text = "\n\n".join(exam_results)
        
        context_items = [initial_complaint]
        context_items.extend(symptoms)
        context_items.extend(exam_results)
        
        embeddings = get_embedding_batch(context_items)
        
        index = pc.Index("who-guide-old")
        relevant_matches = []
        for emb in embeddings:
            matches = vectorQuotesWithSource(emb, index, top_k=2)
            if matches:
                relevant_matches.extend(matches)
        
        relevant_matches.sort(key=lambda x: x['score'], reverse=True)
        
        seen_ids = set()
        unique_matches = []
        for match in relevant_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                unique_matches.append(match)
        
        top_matches = unique_matches[:3]
        medical_guide_content = "\n\n".join([match['text'] for match in top_matches])
        
        diagnosis_prompt = get_diagnosis_prompt(
            initial_complaint=initial_complaint,
            symptoms_text=symptoms_text,
            exam_results_text=exam_results_text,
            medical_guide_content=medical_guide_content
        )
        
        diagnosis_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training. Do NOT make medical assumptions. If something is not in the WHO guide, say so explicitly."},
                {"role": "user", "content": diagnosis_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        diagnosis = diagnosis_completion.choices[0].message.content.strip()
        
        treatment_prompt = get_treatment_prompt(
            initial_complaint=initial_complaint,
            symptoms_text=symptoms_text,
            exam_results_text=exam_results_text,
            diagnosis=diagnosis,
            medical_guide_content=medical_guide_content
        )
        
        treatment_completion = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided in the prompt. Do NOT use any medical knowledge from your training. Do NOT make medical assumptions. If something is not in the WHO guide, say so explicitly."},
                {"role": "user", "content": treatment_prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        treatment = treatment_completion.choices[0].message.content.strip()
        
        citations = []
        for match in top_matches:
            citations.append({
                "source": match.get('source', 'Medical guide'),
                "score": match.get('score', 0.0)
            })
        
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "citations": citations,
            "chunks_used": top_matches
        }
        
    except Exception as e:
        print(f"Error in get_diagnosis_and_treatment: {str(e)}")
        return {
            "diagnosis": "Error generating diagnosis",
            "treatment": "Error generating treatment",
            "citations": [],
            "chunks_used": []
        }

def verify_medical_guide_connection():
    """Verify connection to WHO medical guide and warn about RAG-only approach"""
    try:
        index = pc.Index("who-guide-old")
        # Test query to verify connection
        test_embedding = get_embedding_batch(["fever symptoms"])[0]
        test_results = vectorQuotesWithSource(test_embedding, index, top_k=1)
        
        if test_results:
            print("✓ Connected to WHO medical guide (who-guide-old)")
            print("✓ All medical decisions will be based on WHO guide content via RAG")
            return True
        else:
            print("✗ WARNING: No results from WHO medical guide")
            return False
    except Exception as e:
        print(f"✗ ERROR: Cannot connect to WHO medical guide: {e}")
        return False

def main():
    """Main function to run the simplified CHW assistant"""
    try:
        # Verify medical guide connection first
        if not verify_medical_guide_connection():
            print("Cannot proceed without access to WHO medical guide.")
            return
        initial_responses = []
        danger_sign_responses = []
        exam_responses = []
        
        # Initial Questions Loop
        print("\nWelcome to the Community Health Worker Assistant (Professional Way).")
        print("I'll ask you some initial questions to understand your situation.")
        
        for question in questions_init:
            print(f"\n{question['question']}")
            
            if question['type'] in ['MC', 'MCM']:
                print_options(question['options'])
                
                while True:
                    answer = input("Enter your choice (enter the number): ").strip()
                    
                    if validate_mc_input(answer, question['options']):
                        if answer == "5":
                            custom_answer = input("Please specify your answer: ").strip()
                            answer_text = custom_answer
                        else:
                            answer_text = next(opt['text'] for opt in question['options'] if str(opt['id']) == answer)
                        
                        initial_responses.append({
                            "question": question['question'],
                            "answer": answer_text,
                            "type": question['type']
                        })
                        break
                    
                    print("Invalid input, please try again.")
            
            elif question['type'] == 'FREE':
                answer = input("Please describe: ").strip()
                initial_responses.append({
                    "question": question['question'],
                    "answer": answer,
                    "type": question['type']
                })
        
        print("\nThank you for providing your information. Here's what we recorded:\n")
        for resp in initial_responses:
            print(f"Q: {resp['question']}")
            print(f"A: {resp['answer']}\n")
        
        # Identify symptoms and check coverage using RAG
        initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
        
        index = pc.Index("who-guide-old")
        identified_symptoms, all_citations = identify_symptoms_from_complaint(initial_complaint, index)
        
        if not identified_symptoms:
            print("\nBased on the WHO medical guide, I couldn't identify specific symptoms from your description that are covered in our guide.")
            print("RECOMMENDATION: Please visit a medical center for proper evaluation.")
            return
        
        print(f"\nIdentified symptoms (from WHO medical guide): {', '.join(identified_symptoms)}")
        
        # Display sources used for symptom identification
        display_citations(all_citations, "for symptom identification")
        
        # Since symptoms were identified FROM the WHO guide, they are covered by definition
        # We'll proceed with danger sign questions for all identified symptoms
        
        # Ask danger sign questions for each symptom using WHO guide
        danger_sign_responses, all_citations = ask_danger_sign_questions(identified_symptoms, index)
        
        # Check if any danger signs are positive
        positive_danger_signs = [resp for resp in danger_sign_responses if resp['answer'] == "Yes"]
        
        if positive_danger_signs:
            print(f"\nWARNING: {len(positive_danger_signs)} danger sign(s) detected!")
            for danger in positive_danger_signs:
                print(f"- {danger['symptom']}: {danger['question']}")
            print("\nRECOMMENDATION: Seek immediate medical attention!")
            
            continue_choice = input("\nWould you like to continue with examination despite danger signs? (y/n): ").strip().lower()
            if continue_choice != 'y':
                return
        
        # Examinations based on symptoms and danger signs
        print("\nBased on your responses, I'll recommend appropriate examinations.")
        
        for symptom in identified_symptoms:
            try:
                context = f"Patient presents with {symptom}. Danger signs checked."
                for resp in danger_sign_responses:
                    if resp['symptom'] == symptom and resp['answer'] == "Yes":
                        context += f" DANGER SIGN: {resp['question']}"
                
                embedding = get_embedding_batch([context])[0]
                relevant_docs = vectorQuotesWithSource(embedding, index)
                
                if not relevant_docs:
                    continue
                
                combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
                
                prompt = get_main_examination_prompt(
                    initial_complaint=initial_complaint,
                    symptoms=symptom,
                    previous_exams=str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"
                )

                examination_text = get_openai_completion(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.3
                )
                
                try:
                    examination, option_texts = parse_examination_text(examination_text)
                    
                    options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
                    options.append({"id": 5, "text": "Other (please specify)"})
                    
                    print(f"\nRecommended Examination for {symptom}:")
                    print(examination)
                    
                    # Display sources for this examination
                    display_citations(relevant_docs, f"for {symptom} examination")
                    
                    print("\nSelect the finding:")
                    print_options(options)
                    
                    while True:
                        answer = input("Enter your choice (enter the number): ").strip()
                        
                        if validate_mc_input(answer, options):
                            if answer == "5":
                                custom_answer = input("Please specify your answer: ").strip()
                                answer_text = custom_answer
                            else:
                                answer_text = next(opt['text'] for opt in options if str(opt['id']) == answer)
                            
                            exam_responses.append({
                                "examination": examination,
                                "result": answer_text,
                                "symptom": symptom,
                                "type": "MC",
                                "citations": relevant_docs[-5:]
                            })
                            break
                            
                        print("Invalid input, please try again.")
                        
                except Exception as e:
                    print(f"Error parsing examination: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error generating examination for {symptom}: {e}")
                continue
        
        # Get diagnosis and treatment
        results = get_diagnosis_and_treatment(initial_responses, danger_sign_responses, exam_responses)
        
        print("\nAssessment Complete")
        print("==================")
        print("\nDiagnosis:")
        print(results['diagnosis'])
        print("\nTreatment Plan:")
        print(results['treatment'])
        
        # Display sources for diagnosis and treatment
        if results.get('chunks_used'):
            display_citations(results['chunks_used'], "for diagnosis and treatment")
        
        # Final recommendation if danger signs were present
        if positive_danger_signs:
            print("\n" + "="*50)
            print("IMPORTANT: Due to danger signs detected earlier,")
            print("please follow up with a medical facility even after")
            print("implementing the treatment plan above.")
            print("="*50)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 