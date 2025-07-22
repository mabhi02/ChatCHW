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

def identify_danger_signs(patient_complaint: str, patient_age: int = 30, patient_sex: str = 'Unknown'):
    """
    Identify specific danger signs to look for based on patient symptoms.
    Simple function that asks LLM to identify danger signs.
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
        index = pc.Index("who-guide")
        
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
    # Test cases
    test_cases = [
        {
            "complaint": "I have fever and diarrhea",
            "age": 45,
            "sex": "Female"
        },
        {
            "complaint": "I have a bad headache and feel dizzy",
            "age": 32,
            "sex": "Male"
        },
        {
            "complaint": "I've been coughing for days and feel tired",
            "age": 67,
            "sex": "Female"
        }
    ]
    
    print("=== DANGER SIGNS IDENTIFICATION & RAG QUESTION GENERATION ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST CASE {i}:")
        print(f"Patient: {test_case['age']} year old {test_case['sex']}")
        print(f"Complaint: \"{test_case['complaint']}\"\n")
        
        # Step 1: Identify danger signs
        print("STEP 1: Identifying danger signs...")
        danger_result = identify_danger_signs(
            patient_complaint=test_case['complaint'],
            patient_age=test_case['age'],
            patient_sex=test_case['sex']
        )
        
        if danger_result['status'] == 'success':
            print("✓ Danger signs identified:")
            print()
            for j, danger_sign in enumerate(danger_result['danger_signs'], 1):
                print(f"   • {danger_sign['danger_sign']}")
                print()
            
            # Step 2: Generate RAG questions for ALL danger signs
            print("STEP 2: Generating RAG questions for each danger sign...")
            print()
            
            for k, danger_sign in enumerate(danger_result['danger_signs'], 1):
                print(f"   Question {k}: {danger_sign['danger_sign']}")
                
                # Generate question for this specific danger sign
                single_danger_question = generate_danger_question(
                    danger_signs=[danger_sign],  # Pass single danger sign
                    patient_info=danger_result['patient_info']
                )
                
                if single_danger_question['status'] == 'success':
                    print(f"   ✓ {single_danger_question['question']['text']}")
                    print()
                    print("     Answer choices:")
                    for option in single_danger_question['question']['options']:
                        print(f"        {option['id']}. {option['text']}")
                    print(f"     📚 Sources used: {single_danger_question['metadata']['sources_used']}")
                else:
                    print(f"   ✗ {single_danger_question['message']}")
                
                print()
                print("   " + "-"*50)
                print()
                
        else:
            print(f"✗ Danger sign identification failed: {danger_result['message']}")
        
        print("\n" + "="*80 + "\n") 