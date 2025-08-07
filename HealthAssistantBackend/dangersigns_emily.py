from typing import Dict, List, Any, Optional
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai
import random
import json

# Load environment variables
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's embedding model"""
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [embedding['embedding'] for embedding in response['data']]

def vectorQuotesWithSource(embedding: List[float], index, top_k: int = 5) -> List[Dict]:
    """Query Pinecone index with embedding and return results with source"""
    try:
        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in response['matches']:
            results.append({
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'source': match['metadata'].get('source', 'Unknown')
            })
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def identify_danger_signs(complaint: str, symptoms: List[str], index) -> Dict[str, Any]:
    """
    Identify danger signs from patient complaint and symptoms using RAG.
    Improved to be more relevant and prevent asking irrelevant questions.
    """
    try:
        print(f"Identifying danger signs from complaint: '{complaint}' and symptoms: {symptoms}")
        
        # Create a focused query based on the actual symptoms
        if symptoms:
            # Use the identified symptoms to create a focused query
            symptom_query = " ".join(symptoms)
            query = f"danger signs for: {symptom_query}"
        else:
            # Fallback to complaint if no symptoms identified
            query = f"danger signs for: {complaint}"
        
        print(f"Using focused query: {query}")
        
        # Get embedding for the focused query
        embedding = get_embedding_batch([query])[0]
        relevant_docs = vectorQuotesWithSource(embedding, index, top_k=3)  # Reduced from 5 to 3
        
        print(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            print("No relevant documents found")
            return {
                "status": "success", 
                "danger_signs": [],
                "citations": []
            }
        
        # Use medical guide content to identify danger signs
        medical_guide_content = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Create a more focused prompt that emphasizes relevance
        danger_sign_prompt = f"""
        CRITICAL: You are a WHO Community Health Worker assistant. You can ONLY use information from the WHO medical guide provided below. Do NOT use any medical knowledge from your training.
        
        Patient complaint: "{complaint}"
        Identified symptoms: {symptoms}
        
        Based ONLY on the WHO medical guide content below, identify the MOST RELEVANT danger signs that are:
        1. Specifically related to the patient's symptoms: {symptoms}
        2. Explicitly mentioned in the WHO medical guide content above
        3. Most critical for immediate assessment
        
        WHO Medical Guide Content:
        {medical_guide_content}
        
        IMPORTANT RULES:
        - ONLY identify danger signs that are DIRECTLY relevant to the patient's symptoms
        - If the patient has fever/chills, focus on fever-related danger signs
        - If the patient has cough, focus on respiratory danger signs
        - If the patient has diarrhea, focus on diarrhea-related danger signs
        - Do NOT identify danger signs that are unrelated to the patient's symptoms
        - Limit to 2-3 most critical danger signs maximum
        
        Return ONLY the specific danger signs that are relevant to the patient's symptoms, formatted as a simple comma-separated list (e.g., "chest indrawing, fast breathing").
        If no relevant danger signs can be identified from the WHO medical guide, respond with "none"
        
        Remember: Use ONLY information from the WHO guide above, not your medical training. Focus on relevance to the patient's specific symptoms.
        """
        
        print("Sending danger sign identification prompt to OpenAI")
        response = get_openai_completion(
            danger_sign_prompt,
            max_tokens=100,
            temperature=0.3
        )
        
        print(f"OpenAI response: {response}")
        
        # Parse the response to extract danger signs
        if response.lower().strip() == "none":
            print("No danger signs identified")
            return {
                "status": "success", 
                "danger_signs": [],
                "citations": relevant_docs
            }
        
        # Parse danger signs from response
        danger_signs = [s.strip() for s in response.split(',')]
        danger_signs = [s for s in danger_signs if s and len(s) > 2]
        
        # LIMIT: Only take the first 3 most relevant danger signs
        MAX_DANGER_SIGNS = 3
        danger_signs = danger_signs[:MAX_DANGER_SIGNS]
        
        print(f"Parsed danger signs (limited to {MAX_DANGER_SIGNS}): {danger_signs}")
        
        # Convert to structured format
        final_danger_signs = []
        seen_signs = set()

        for sign in danger_signs:
            clean_sign = sign.strip()
            if clean_sign and clean_sign.lower() not in seen_signs:
                seen_signs.add(clean_sign.lower())
                final_danger_signs.append({
                    "danger_sign": clean_sign,
                    "priority": "HIGH"
                })

        # LIMIT: Only return the first 3 most relevant danger signs
        MAX_DANGER_SIGNS = 3
        final_danger_signs = final_danger_signs[:MAX_DANGER_SIGNS]
        
        print(f"Final cleaned danger signs (limited to {MAX_DANGER_SIGNS}): {len(final_danger_signs)}")
        for sign in final_danger_signs:
            print(f"  - {sign['danger_sign']}")
        
        return {
            "status": "success", 
            "danger_signs": final_danger_signs,
            "citations": relevant_docs
        }
        
    except Exception as e:
        print(f"Error in identify_danger_signs: {str(e)}")
        return {
            "status": "error",
            "danger_signs": [],
            "citations": []
    }


# the previous identify_danger_signs function
def identify_danger_signs_OLD(patient_complaint: str, patient_age: int = 30, patient_sex: str = 'Unknown'):
    """
    CURRENT APPROACH (AI Memory):
    - Uses AI's medical training to generate danger signs
    - Prompt asks AI to act like "expert emergency physician"
    - No RAG retrieval happening here
    
    EMILY'S TASK:
    - Change this to use RAG FIRST
    - Search WHO guidelines for patient complaint
    - Extract danger signs mentioned IN the guidelines
    - Don't let AI generate from memory
    """
    try:
        if not patient_complaint:
            return {
                "status": "error",
                "message": "Patient complaint is required"
            }
        
        prompt = f'''You are an expert emergency physician. A {patient_age} year old {patient_sex} presents with: "{patient_complaint}"

Based on your extensive clinical experience treating thousands of similar cases, immediately identify the most critical danger signs that must be urgently assessed. Remember that early detection saves lives, so err on the side of caution and include all potentially serious warning signs.

These danger signs are often subtle and can be easily missed, but when present indicate immediate life-threatening conditions that require emergency intervention. Think about the worst-case scenarios and most aggressive presentations you've seen.

List exactly 3-5 specific danger signs to assess, formatted as a simple list. Be very specific about what to look for (like "blood in stool" not just "GI bleeding"). Focus on the most dramatic and concerning signs first.

Critical danger signs to assess:'''
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Parse the response into danger signs
        danger_signs_text = response.choices[0].message.content.strip()
        danger_signs_list = []
        
        for line in danger_signs_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('Critical danger'):
                # Remove numbering/bullets
                if line[0].isdigit() or line.startswith('•') or line.startswith('-'):
                    line = line[1:].strip()
                    if line.startswith('.') or line.startswith('-') or line.startswith(')'):
                        line = line[1:].strip()
                
                if line:
                    danger_signs_list.append({
                        "danger_sign": line,
                        "priority": random.choice(["HIGH", "MODERATE", "LOW"])
                    })
        
        return {
            "status": "success",
            "danger_signs": danger_signs_list,
            "patient_info": {
                "complaint": patient_complaint,
                "age": patient_age,
                "sex": patient_sex
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def generate_danger_question(danger_signs: List[Dict], patient_info: Dict):
    """
    Generate a high-quality follow-up question based on identified danger signs.
    This uses excellent RAG with the medical guide to create precise questions.
    """
    try:
        if not danger_signs:
            return {
                "status": "error", 
                "message": "No danger signs provided"
            }
        
        # Select the highest priority danger sign to focus on
        priority_order = {"HIGH": 3, "MODERATE": 2, "LOW": 1}
        primary_danger_sign = max(danger_signs, key=lambda x: priority_order.get(x.get('priority', 'LOW'), 1))
        
        danger_sign_name = primary_danger_sign.get('danger_sign', '')
        
        # Extract just the core terms from the long danger sign description
        core_danger_sign = danger_sign_name.lower()
        
        # Remove explanatory text after common separators
        separators = [' - ', ': ', ' (', ' -', ' indicating', ' suggesting', ' which', ' that', ',', ' e.g.', ' especially']
        
        for sep in separators:
            if sep in core_danger_sign:
                core_danger_sign = core_danger_sign.split(sep)[0].strip()
                break
        
        # Clean up common prefixes/suffixes and punctuation
        core_danger_sign = core_danger_sign.replace('signs of ', '').replace('sign of ', '')
        core_danger_sign = core_danger_sign.replace('severe ', '').replace('persistent ', '')
        core_danger_sign = core_danger_sign.strip('.,;:!?')
        
        # Handle specific medical term mappings
        if 'hypotension' in core_danger_sign or 'shock' in core_danger_sign:
            core_danger_sign = 'shock'
        elif 'blood in stool' in core_danger_sign:
            core_danger_sign = 'blood in stool'
        elif 'mental status' in core_danger_sign or 'confusion' in core_danger_sign:
            core_danger_sign = 'confusion'
        elif 'abdominal pain' in core_danger_sign:
            core_danger_sign = 'abdominal pain'
        elif 'dehydration' in core_danger_sign:
            core_danger_sign = 'dehydration'
        elif 'neck stiff' in core_danger_sign:
            core_danger_sign = 'neck stiffness'
        elif 'headache' in core_danger_sign:
            core_danger_sign = 'headache'
        elif 'cough' in core_danger_sign and 'blood' in core_danger_sign:
            core_danger_sign = 'coughing blood'
        elif 'breathing' in core_danger_sign or 'respiratory' in core_danger_sign:
            core_danger_sign = 'difficulty breathing'
        elif 'chest pain' in core_danger_sign:
            core_danger_sign = 'chest pain'
        elif 'fever' in core_danger_sign:
            core_danger_sign = 'fever'
        
        print(f"   Searching for: '{core_danger_sign}' (from: '{danger_sign_name[:50]}...')")
        
        # Do a specific vector search for the core danger sign
        index = pc.Index("who-guide-old")
        
        # Search specifically for the core danger sign terms
        search_queries = [
            core_danger_sign,
            f"{core_danger_sign} assessment",
            f"check for {core_danger_sign}",
            f"{core_danger_sign} questions"
        ]
        
        all_relevant_docs = []
        for query in search_queries:
            embedding = get_embedding_batch([query])[0]
            docs = vectorQuotesWithSource(embedding, index, top_k=2)  # Reduced from 3 to 2
            all_relevant_docs.extend(docs)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_docs = []
        for doc in all_relevant_docs:
            if doc['id'] not in seen_ids:
                seen_ids.add(doc['id'])
                unique_docs.append(doc)
        
        # Sort by relevance score and take top results
        unique_docs.sort(key=lambda x: x['score'], reverse=True)
        relevant_docs = unique_docs[:2]  # Use only top 2 most relevant
        
        if not relevant_docs:
            return {
                "status": "error",
                "message": f"Can't find '{danger_sign_name}' in the medical guide - no relevant information available"
            }
        
        # Format medical guide information from vector search results
        medical_guide_content = "\n\n---Document Separator---\n\n".join([
            f"Source: {doc['source']}\nContent: {doc['text']}" 
            for doc in relevant_docs
        ])


        # Use the exact same approach as the main RAG system
        prompt = f'''Danger sign to assess: "{core_danger_sign}"

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. The question must be based directly on assessment questions mentioned in the medical guide.
4. If the medical guide mentions specific questions to ask for this condition, use those exact questions.
5. Focus on gathering information about "{core_danger_sign}".
6. Return only the question text without any explanation.
7. Make the question simple and easy for a health worker to ask.

Based ONLY on the medical guide information, what question should be asked to assess "{core_danger_sign}" in this patient?
'''
        
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        question = response.choices[0].message.content.strip()
        
        # Generate options using the exact same approach as the main system
        options_prompt = f'''QUESTION: "{question}"

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

INSTRUCTIONS:
1. ONLY use information from the medical guide above
2. Generate 4 answer options for the question
3. If the medical guide mentions specific answer options, use those exactly
4. If not, create relevant options based only on the medical guide content
5. Format as a numbered list (1-4)
6. Do not include explanations, only the options themselves

Generate 4 answer options:'''
        
        options_response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."},
                {"role": "user", "content": options_prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        # Parse options
        options = []
        for i, opt in enumerate(options_response.choices[0].message.content.strip().split('\n')):
            if opt.strip():
                text = opt.strip()
                # Remove numbering if present
                if len(text) > 2 and text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                elif len(text) > 3 and text[:2].isdigit() and text[2] in ['.','-',')']:
                    text = text[3:].strip()
                
                if text:  # Only add non-empty options
                    options.append({"id": len(options) + 1, "text": text})
                    
                if len(options) >= 4:  # Limit to 4 options
                    break
        
        # Add "Other" option
        options.append({"id": len(options) + 1, "text": "Other (please specify)"})
        
        # Format comprehensive response
        return {
            "status": "success",
            "question": {
                "text": question,
                "options": options,
                "type": "MC"
            },
            "metadata": {
                "danger_sign_assessed": danger_sign_name,
                "core_search_term": core_danger_sign,
                "sources_used": len(relevant_docs),
                "source_details": [
                    {
                        "source": doc["source"], 
                        "relevance_score": round(doc["score"], 3)
                    } 
                    for doc in relevant_docs
                ],
                "rag_only": True,
                "medical_guide_only": True
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Question generation failed: {str(e)}"
        }

if __name__ == '__main__':
    print("=== COMPARING RAG vs AI-MEMORY APPROACHES ===\n")
    
    test_cases = [
        {"complaint": "fever and diarrhea", "age": 5, "sex": "Female"},
        {"complaint": "headache", "age": 32, "sex": "Male"},
        {"complaint": "cough", "age": 3, "sex": "Male"}
    ]
    
    for test_case in test_cases:
        print(f"Patient: {test_case['age']}yo {test_case['sex']} with '{test_case['complaint']}'")
        print("="*60)
        
        # Your new RAG approach
        print("\n RAG-FIRST APPROACH:")
        rag_result = identify_danger_signs(
            test_case['complaint'], 
            test_case['age'], 
            test_case['sex']
        )
        
        # Old AI-memory approach  
        print("\n AI-MEMORY APPROACH:")
        ai_result = identify_danger_signs_OLD(
            test_case['complaint'],
            test_case['age'], 
            test_case['sex']
        )
        
        print(f"\nRAG found: {len(rag_result['danger_signs'])} danger signs")
        print(f"AI found: {len(ai_result['danger_signs'])} danger signs")
        
        print("\n" + "="*80 + "\n")

# OLD MAIN
# if __name__ == '__main__':

#     # # Add your test
#     # print("\n" + "="*50)
#     # print("EMILY'S RAG DANGER SIGNS TEST")
#     # print("="*50)
    
#     # # Test cases
#     # test_cases = [
#     #     ("fever and diarrhea", 5, "Female"),
#     #     ("headache", 25, "Male"),
#     #     ("cough", 3, "Male")
#     # ]
    
#     # for complaint, age, sex in test_cases:
#     #     print(f"\nTesting: {age}yo {sex} with '{complaint}'")
#     #     print("-" * 40)
        
#     #     result = identify_danger_signs_from_guidelines(complaint, age, sex)
#     #     print(f"Status: {result['status']}")
#     #     print(f"Danger signs found: {result['danger_signs']}")
        
#     #     # This will show your search results for debugging
#     #     print()

#     # Search the existing database to see what's there
#     index = pc.Index("who-guide-old")
#     embedding = get_embedding_batch(["fever danger signs"])[0]
#     results = vectorQuotesWithSource(embedding, index, top_k=10)

#     for result in results:
#         print(f"Source: {result['source']}")
#         print(f"Text: {result['text'][:200]}...")
#         print("---")
#     # Test cases
#     test_cases = [
#         {
#             "complaint": "I have fever and diarrhea",
#             "age": 45,
#             "sex": "Female"
#         },
#         {
#             "complaint": "I have a bad headache and feel dizzy",
#             "age": 32,
#             "sex": "Male"
#         },
#         {
#             "complaint": "I've been coughing for days and feel tired",
#             "age": 67,
#             "sex": "Female"
#         }
#     ]
    
#     print("=== DANGER SIGNS IDENTIFICATION & RAG QUESTION GENERATION ===\n")
    
#     for i, test_case in enumerate(test_cases, 1):
#         print(f"TEST CASE {i}:")
#         print(f"Patient: {test_case['age']} year old {test_case['sex']}")
#         print(f"Complaint: \"{test_case['complaint']}\"\n")
        
#         # Step 1: Identify danger signs
#         print("STEP 1: Identifying danger signs...")
#         danger_result = identify_danger_signs(
#             patient_complaint=test_case['complaint'],
#             patient_age=test_case['age'],
#             patient_sex=test_case['sex']
#         )
        
#         if danger_result['status'] == 'success':
#             print("✓ Danger signs identified:")
#             print()
#             for j, danger_sign in enumerate(danger_result['danger_signs'], 1):
#                 print(f"   • {danger_sign['danger_sign']}")
#                 print()
            
#             # Step 2: Generate RAG questions for ALL danger signs
#             print("STEP 2: Generating RAG questions for each danger sign...")
#             print()
            
#             for k, danger_sign in enumerate(danger_result['danger_signs'], 1):
#                 print(f"   Question {k}: {danger_sign['danger_sign']}")
                
#                 # Generate question for this specific danger sign
#                 single_danger_question = generate_danger_question(
#                     danger_signs=[danger_sign],  # Pass single danger sign
#                     patient_info=danger_result['patient_info']
#                 )
                
#                 if single_danger_question['status'] == 'success':
#                     print(f"   ✓ {single_danger_question['question']['text']}")
#                     print()
#                     print("     Answer choices:")
#                     for option in single_danger_question['question']['options']:
#                         print(f"        {option['id']}. {option['text']}")
#                     print(f"     📚 Sources used: {single_danger_question['metadata']['sources_used']}")
#                 else:
#                     print(f"   ✗ {single_danger_question['message']}")
                
#                 print()
#                 print("   " + "-"*50)
#                 print()
                
#         else:
#             print(f"✗ Danger sign identification failed: {danger_result['message']}")
        
#         print("\n" + "="*80 + "\n") 


#         # print("\n" + "="*50)
#         # print("=== EMILY'S RAG COMPARISON TEST ===")
#         # print("="*50)
        
#         # test_complaint = "fever and diarrhea"
        
#         # print("ORIGINAL:")
#         # original = identify_danger_signs(test_complaint, 45, "Female")
#         # print(original)
        
#         # print("\nNEW (TODO):")
#         # new = identify_danger_signs_from_guidelines(test_complaint, 45, "Female")
#         # print(new)