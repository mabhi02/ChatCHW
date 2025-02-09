from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
import torch
import torch.nn as nn
import json

from AVM.MATRIX.matrix_core import MATRIX
from AVM.MATRIX.decoder_tuner import DecoderTuner
from AVM.MATRIX.attention_viz import AttentionVisualizer
from AVM.MATRIX.state_encoder import StateSpaceEncoder
from AVM.MATRIX.pattern_analyzer import PatternAnalyzer
from AVM.MATRIX.config import MATRIXConfig

# Load environment variables and initialize clients
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize MATRIX system and components
matrix = MATRIX()
decoder_tuner = DecoderTuner(matrix.meta_learner.decoder)
visualizer = AttentionVisualizer()
pattern_analyzer = PatternAnalyzer()

# Initial screening questions
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

def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI's API."""
    try:
        response = openai.Embedding.create(
            input=texts,
            engine="text-embedding-3-small"
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
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
        return []

def compress_medical_context(responses: List[Dict[str, Any]], 
                           embeddings: Optional[List[List[float]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Compress medical context by using embeddings to find key information."""
    text_chunks = []
    for resp in responses:
        if isinstance(resp.get('answer'), str):
            text_chunks.append(f"{resp['question']}: {resp['answer']}")
        elif isinstance(resp.get('answer'), list):
            text_chunks.append(f"{resp['question']}: {', '.join(resp['answer'])}")
    
    if not embeddings:
        embeddings = get_embedding_batch(text_chunks)
    
    compressed_chunks = []
    seen_content = set()
    
    for chunk, embedding in zip(text_chunks, embeddings):
        if chunk not in seen_content:
            compressed_chunks.append(chunk)
            seen_content.add(chunk)
    
    return "\n".join(compressed_chunks[:5]), []

def get_formatted_options(options: List[Dict[str, Any]]) -> List[str]:
    """Return formatted options for multiple choice questions."""
    return [f"{option['id']}: {option['text']}" for option in options]

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

def get_initial_responses(responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process initial screening questions with given responses."""
    processed_responses = []
    current_response_idx = 0
    
    for question in questions_init:
        if question['type'] in ['MC', 'YN', 'MCM']:
            if current_response_idx < len(responses):
                answer = responses[current_response_idx]
                if question['type'] == 'MCM':
                    if isinstance(answer, list):
                        processed_responses.append({
                            "question": question['question'],
                            "answer": answer,
                            "type": question['type']
                        })
                else:
                    processed_responses.append({
                            "question": question['question'],
                            "answer": answer,
                            "type": question['type']
                        })
                current_response_idx += 1
                
        elif question['type'] == 'NUM':
            if current_response_idx < len(responses):
                answer = responses[current_response_idx]
                if validated_num := validate_num_input(str(answer), question['range']):
                    processed_responses.append({
                        "question": question['question'],
                        "answer": validated_num,
                        "type": question['type']
                    })
                current_response_idx += 1
                
        elif question['type'] == 'FREE':
            if current_response_idx < len(responses):
                answer = responses[current_response_idx]
                if answer:
                    processed_responses.append({
                        "question": question['question'],
                        "answer": answer,
                        "type": question['type']
                    })
                current_response_idx += 1
    
    return processed_responses

def process_with_matrix(current_text: str, previous_responses: List[Dict], 
                       context_text: str = "") -> Dict[str, Any]:
    """Process input through MATRIX system with enhanced error handling."""
    try:
        patterns = pattern_analyzer.analyze_patterns(current_text)
        pattern_output = {
            "optimist": patterns['optimist_confidence'],
            "pessimist": patterns['pessimist_confidence']
        }
        
        state = matrix.state_encoder.encode_state(
            [],
            previous_responses,
            current_text
        )
        
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if len(state.shape) == 4:
            state = state.squeeze(1)
            
        optimist_view = matrix.optimist.evaluate(
            state,
            context_text if context_text else current_text
        )
        pessimist_view = matrix.pessimist.evaluate(
            state,
            context_text if context_text else current_text
        )
        
        matrix_output = matrix.process_state(
            [], 
            previous_responses,
            current_text
        )
        
        return {
            "pattern_analysis": pattern_output,
            "matrix_output": matrix_output
        }
        
    except Exception as e:
        return {
            "pattern_analysis": {"optimist": 0.5, "pessimist": 0.5},
            "matrix_output": {
                "confidence": 0.5,
                "selected_agent": "optimist",
                "weights": {"optimist": 0.5, "pessimist": 0.5}
            }
        }

def judge(followup_responses: List[Dict[str, Any]], current_question: str) -> Dict[str, Any]:
    """Judge if the current question is too similar using MATRIX."""
    if len(followup_responses) >= MATRIXConfig.MAX_QUESTIONS:
        return {
            "should_stop": True,
            "reason": "max_questions",
            "matrix_output": None
        }
        
    if not followup_responses:
        return {
            "should_stop": False,
            "reason": None,
            "matrix_output": None
        }
        
    try:
        matrix_output = process_with_matrix(
            current_question, 
            followup_responses
        )
        
        should_stop = (
            matrix_output["matrix_output"]["confidence"] > MATRIXConfig.SIMILARITY_THRESHOLD or
            len(followup_responses) >= MATRIXConfig.MAX_FOLLOWUPS or
            matrix_output["matrix_output"]["weights"]["optimist"] > 0.7
        )
        
        return {
            "should_stop": should_stop,
            "reason": "matrix_analysis" if should_stop else None,
            "matrix_output": matrix_output
        }
        
    except Exception as e:
        return {
            "should_stop": len(followup_responses) >= 5,
            "reason": "fallback",
            "matrix_output": None
        }

def judge_exam(previous_exams: List[Dict[str, Any]], current_exam: str) -> Dict[str, Any]:
    """Judge examination similarity with improved duplicate detection."""
    if not previous_exams:
        return {
            "should_stop": False,
            "reason": None,
            "matrix_output": None
        }
        
    try:
        if len(previous_exams) >= MATRIXConfig.MAX_EXAMS:
            return {
                "should_stop": True,
                "reason": "max_exams",
                "matrix_output": None
            }
            
        exam_lines = current_exam.split('\n')
        current_exam_name = ""
        current_procedure = ""
        
        for line in exam_lines:
            if line.startswith("Examination:"):
                current_exam_name = line.split('Examination:')[1].strip().lower()
            elif line.startswith("Procedure:"):
                current_procedure = line.split('Procedure:')[1].strip().lower()
        
        for exam in previous_exams:
            prev_exam_lines = exam['examination'].split('\n')
            prev_name = ""
            prev_procedure = ""
            
            for line in prev_exam_lines:
                if line.startswith("Examination:"):
                    prev_name = line.split('Examination:')[1].strip().lower()
                elif line.startswith("Procedure:"):
                    prev_procedure = line.split('Procedure:')[1].strip().lower()
            
            if (prev_name in current_exam_name or current_exam_name in prev_name):
                return {
                    "should_stop": True,
                    "reason": "similar_exam",
                    "matrix_output": None
                }
                
            if len(prev_procedure) > 0 and len(current_procedure) > 0:
                words1 = set(prev_procedure.split())
                words2 = set(current_procedure.split())
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if similarity > 0.7:
                    return {
                        "should_stop": True,
                        "reason": "similar_procedure",
                        "matrix_output": None
                    }
        
        matrix_output = process_with_matrix(
            current_exam, 
            previous_exams
        )
        
        should_end = (
            matrix_output["matrix_output"]["confidence"] > MATRIXConfig.EXAM_SIMILARITY_THRESHOLD or
            len(previous_exams) >= MATRIXConfig.MAX_EXAMS or
            matrix_output["matrix_output"]["weights"]["optimist"] > 0.8
        )
        
        return {
            "should_stop": should_end,
            "reason": "matrix_analysis" if should_end else None,
            "matrix_output": matrix_output
        }
                
    except Exception as e:
        return {
            "should_stop": len(previous_exams) >= MATRIXConfig.MAX_EXAMS,
            "reason": "fallback",
            "matrix_output": None
        }

def store_question(question_data: dict) -> None:
    """Store a question in the global array."""
    global structured_questions_array
    structured_questions_array.append(question_data)

def parse_question_data(question: str, options: list, answer: str, matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format."""
    global structured_questions_array
    
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    selected_idx = next((i+1 for i, opt in enumerate(options) if opt['text'] == answer), None)
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" for cite in citations] if citations else []
    
    question_data = {
        "question": question,
        "options": formatted_options,
        "selected_option": selected_idx,
        "pattern": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "confidence": matrix_output.get('confidence', 0.5),
        "selected_agent": matrix_output.get('selected_agent', 'optimist'),
        "weights": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "sources": sources
    }
    
    structured_questions_array.append(question_data)
    return question_data

def get_followup_questions(initial_responses: List[Dict[str, Any]], user_input: Optional[str] = None) -> Dict[str, Any]:
    """Generate and handle follow-up questions with enhanced processing and citations."""
    followup_responses = []
    
    initial_complaint = next((resp['answer'] for resp in initial_responses 
                            if resp['question'] == "Please describe what brings you here today"), "")
    
    output_data = {
        "status": "asking",
        "question": None,
        "options": None,
        "matrix_output": None,
        "citations": None,
        "responses": followup_responses,
        "message": None,
        "requires_input": True
    }

    try:
        context = f"Initial complaint: {initial_complaint}\n"
        if followup_responses:
            context += "Previous responses:\n"
            for resp in followup_responses:
                context += f"Q: {resp['question']}\nA: {resp['answer']}\n"
        
        embedding = get_embedding_batch([context])[0]
        relevant_docs = vectorQuotesWithSource(embedding, pc.Index("final-asha"))
        
        if not relevant_docs:
            output_data["status"] = "error"
            output_data["message"] = "Could not generate relevant question"
            return output_data
        
        combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
        
        previous_questions = "\n".join([f"- {resp['question']}" for resp in followup_responses])
        prompt = f'''Based on the patient's initial complaint: "{initial_complaint}"
        
        Previous questions asked:
        {previous_questions if followup_responses else "No previous questions yet"}
        
        Relevant medical context:
        {combined_context}
        
        Generate ONE focused, relevant follow-up question that is different from the previous questions.
        Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
        Follow standard medical assessment order:
        1. Duration and onset
        2. Characteristics and severity
        3. Associated symptoms
        4. Impact on daily life
        
        Return only the question text.'''
        
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        
        question = completion.choices[0].message.content.strip()
        
        # Generate options
        options_prompt = f'''Generate 4 concise answers for: "{question}"
        Clear, mutually exclusive options.
        Return each option on a new line (1-4).'''
        
        options_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": options_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=100
        )
        
        options = []
        for i, opt in enumerate(options_completion.choices[0].message.content.strip().split('\n')):
            if opt.strip():
                text = opt.strip()
                if text[0].isdigit() and text[1] in ['.','-',')']:
                    text = text[2:].strip()
                options.append({"id": i+1, "text": text})
        
        options.append({"id": 5, "text": "Other (please specify)"})
        
        # Use MATRIX to judge similarity and get pattern analysis
        matrix_output = process_with_matrix(question, followup_responses)
        
        # If we have user input, process it
        if user_input:
            if validate_mc_input(user_input, options):
                if user_input == "5":
                    return {
                        "status": "needs_custom",
                        "question": question,
                        "options": options,
                        "matrix_output": matrix_output,
                        "citations": relevant_docs[-5:],
                        "responses": followup_responses,
                        "requires_input": True
                    }
                else:
                    answer_text = next(opt['text'] for opt in options if str(opt['id']) == user_input)
                    
                    # Store response
                    followup_responses.append({
                        "question": question,
                        "answer": answer_text,
                        "type": "MC",
                        "citations": relevant_docs[-5:]
                    })
                    
                    # Parse and store question data
                    parse_question_data(
                        question, 
                        options, 
                        answer_text, 
                        matrix_output["matrix_output"], 
                        relevant_docs[-5:]
                    )
                    
                    judge_result = judge(followup_responses, question)
                    
                    return {
                        "status": "complete" if judge_result["should_stop"] else "next_question",
                        "question": question,
                        "options": options,
                        "matrix_output": matrix_output,
                        "citations": relevant_docs[-5:],
                        "responses": followup_responses,
                        "requires_input": not judge_result["should_stop"]
                    }
            else:
                return {
                    "status": "invalid_input",
                    "question": question,
                    "options": options,
                    "matrix_output": matrix_output,
                    "citations": relevant_docs[-5:],
                    "responses": followup_responses,
                    "message": "Invalid input, please try again",
                    "requires_input": True
                }
        
        # Return current question state if no input
        return {
            "status": "asking",
            "question": question,
            "options": options,
            "matrix_output": matrix_output,
            "citations": relevant_docs[-5:],
            "responses": followup_responses,
            "requires_input": True
        }
                
    except Exception as e:
        output_data["status"] = "error"
        output_data["message"] = f"Error generating follow-up question: {str(e)}"
        return output_data
    
def parse_examination_text(examination_text: str) -> tuple[str, list[str]]:
   """
   Parse examination text using "#:" delimiter.
   Returns tuple of (examination text, list of options).
   """
   parts = examination_text.split("#:")
   
   if len(parts) < 2:
       raise ValueError("Invalid examination format - missing '#:' delimiter")
       
   examination = parts[0].strip()
   options = [opt.strip() for opt in parts[1:] if opt.strip()]
   
   return examination, options

def store_examination(examination_text: str, selected_option: int) -> Dict[str, Any]:
   """Store examination data in the global examination history."""
   global examination_history
   
   try:
       examination, options = parse_examination_text(examination_text)
       
       examination_entry = {
           "examination": examination,
           "options": options,
           "selected_option": selected_option
       }
       
       examination_history.append(examination_entry)
       return {"status": "success", "data": examination_entry}
       
   except Exception as e:
       return {"status": "error", "message": f"Error storing examination: {str(e)}"}

def get_followup_exams(initial_responses: List[Dict[str, Any]], 
                     followup_responses: List[Dict[str, Any]], 
                     user_input: Optional[str] = None) -> Dict[str, Any]:
   """Generate and handle examinations with citations."""
   exam_responses = []
   initial_complaint = next((resp['answer'] for resp in initial_responses 
                           if resp['question'] == "Please describe what brings you here today"), "")
   
   output_data = {
       "status": "asking",
       "examination": None,
       "options": None,
       "citations": None,
       "responses": exam_responses,
       "message": None,
       "requires_input": True
   }
   
   index = pc.Index("final-asha")
   exam_citations = []
   
   # Track symptoms
   symptoms = set()
   for resp in followup_responses:
       answer = resp['answer'].lower()
       for symptom in ['pain', 'fever', 'cough', 'fatigue', 'weakness', 'swelling', 
                      'headache', 'nausea', 'dizziness', 'rash']:
           if symptom in answer:
               symptoms.add(symptom)
   
   try:
       context = f"""Initial complaint: {initial_complaint}
Key symptoms: {', '.join(symptoms)}
Previous findings: {str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"}"""

       embedding = get_embedding_batch([context])[0]
       relevant_docs = vectorQuotesWithSource(embedding, index)
       
       if not relevant_docs:
           output_data["status"] = "error"
           output_data["message"] = "Could not generate relevant examination"
           return output_data
       
       exam_citations.extend(relevant_docs)
       combined_context = " ".join([doc["text"] for doc in relevant_docs[:2]])
       
       prompt = f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {', '.join(symptoms)}

Previous exams: {str([exam['examination'] for exam in exam_responses]) if exam_responses else "None"}

Recommend ONE essential examination in this format (should not be first world exams like MRI, CT, Colonoscopy, etc.):
[Examination name and procedure]
#:[First possible finding]
#:[Second possible finding]
#:[Third possible finding]
#:[Fourth possible finding]'''

       completion = groq_client.chat.completions.create(
           messages=[{"role": "system", "content": prompt}],
           model="llama-3.3-70b-versatile",
           temperature=0.3,
           max_tokens=200
       )
       
       examination_text = completion.choices[0].message.content.strip()
       
       judge_result = judge_exam(exam_responses, examination_text)
       if judge_result["should_stop"]:
           return {
               "status": "complete",
               "responses": exam_responses,
               "requires_input": False
           }
           
       try:
           examination, option_texts = parse_examination_text(examination_text)
           options = [{"id": i+1, "text": text} for i, text in enumerate(option_texts)]
           options.append({"id": 5, "text": "Other (please specify)"})
           
           if user_input:
               if validate_mc_input(user_input, options):
                   store_result = store_examination(examination_text, int(user_input))
                   if store_result["status"] != "success":
                       return {
                           "status": "error",
                           "message": store_result["message"],
                           "examination": examination,
                           "options": options,
                           "requires_input": True
                       }
                   
                   if user_input == "5":
                       return {
                           "status": "needs_custom",
                           "examination": examination,
                           "options": options,
                           "citations": exam_citations[-5:],
                           "responses": exam_responses,
                           "requires_input": True
                       }
                   else:
                       selected_text = next(opt['text'] for opt in options if str(opt['id']) == user_input)
                       exam_responses.append({
                           "examination": examination,
                           "result": selected_text,
                           "type": "EXAM",
                           "citations": exam_citations[-5:]
                       })
                       
                       judge_result = judge_exam(exam_responses, examination_text)
                       return {
                           "status": "complete" if judge_result["should_stop"] else "next_exam",
                           "examination": examination,
                           "options": options,
                           "citations": exam_citations[-5:],
                           "responses": exam_responses,
                           "requires_input": not judge_result["should_stop"]
                       }
               else:
                   return {
                       "status": "invalid_input",
                       "examination": examination,
                       "options": options,
                       "citations": exam_citations[-5:],
                       "responses": exam_responses,
                       "message": "Invalid input, please try again",
                       "requires_input": True
                   }
           
           return {
               "status": "asking",
               "examination": examination,
               "options": options,
               "citations": exam_citations[-5:],
               "responses": exam_responses,
               "requires_input": True
           }
               
       except ValueError as e:
           output_data["status"] = "error"
           output_data["message"] = f"Error parsing examination: {str(e)}"
           return output_data
               
   except Exception as e:
       output_data["status"] = "error"
       output_data["message"] = f"Error generating examination: {str(e)}"
       return output_data

def get_diagnosis_and_treatment(initial_responses: List[Dict[str, Any]], 
                             followup_responses: List[Dict[str, Any]], 
                             exam_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
   """Get diagnosis and treatment with citations."""
   try:
       initial_complaint = next((resp['answer'] for resp in initial_responses 
                           if resp['question'] == "Please describe what brings you here today"), "")
       
       key_findings = []
       for resp in followup_responses + exam_responses:
           if isinstance(resp.get('answer'), str):
               key_findings.append(f"{resp['answer']}")
       key_findings = key_findings[-3:]
       
       index = pc.Index("final-asha")
       diagnosis_embedding = get_embedding_batch([initial_complaint + " diagnosis"])[0]
       diagnosis_docs = vectorQuotesWithSource(diagnosis_embedding, index, top_k=2)
       
       diagnosis_context = " ".join([doc["text"] for doc in diagnosis_docs])
       short_diagnosis_prompt = f'''Patient complaint: {initial_complaint}
Key findings: {"; ".join(key_findings)}
Reference: {diagnosis_context[:200]}

List top 3-4 possible diagnoses based on symptoms.'''

       diagnosis_completion = groq_client.chat.completions.create(
           messages=[{"role": "system", "content": short_diagnosis_prompt}],
           model="llama-3.3-70b-versatile",
           temperature=0.2,
           max_tokens=100
       )
       
       diagnosis = diagnosis_completion.choices[0].message.content.strip()
       
       treatment_parts = []
       treatment_docs = []
       
       immediate_embedding = get_embedding_batch([initial_complaint + " immediate care steps"])[0]
       immediate_docs = vectorQuotesWithSource(immediate_embedding, index, top_k=1)
       treatment_docs.extend(immediate_docs)
       
       if immediate_docs:
           immediate_prompt = f'''Based on: {immediate_docs[0]["text"][:200]}
Provide 2-3 immediate care steps for {initial_complaint}.'''
           
           immediate_completion = groq_client.chat.completions.create(
               messages=[{"role": "system", "content": immediate_prompt}],
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_tokens=100
           )
           treatment_parts.append("Immediate Care:\n" + immediate_completion.choices[0].message.content.strip())
       
       med_embedding = get_embedding_batch([initial_complaint + " medications treatment"])[0]
       med_docs = vectorQuotesWithSource(med_embedding, index, top_k=1)
       treatment_docs.extend(med_docs)
       
       if med_docs:
           med_prompt = f'''Based on: {med_docs[0]["text"][:200]}
List 2-3 key medications or supplements for {initial_complaint}.'''
           
           med_completion = groq_client.chat.completions.create(
               messages=[{"role": "system", "content": med_prompt}],
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_tokens=100
           )
           treatment_parts.append("\nMedications/Supplements:\n" + med_completion.choices[0].message.content.strip())
       
       home_embedding = get_embedding_batch([initial_complaint + " home care follow up"])[0]
       home_docs = vectorQuotesWithSource(home_embedding, index, top_k=1)
       treatment_docs.extend(home_docs)
       
       if home_docs:
           home_prompt = f'''Based on: {home_docs[0]["text"][:200]}
List 2-3 home care instructions for {initial_complaint}.'''
           
           home_completion = groq_client.chat.completions.create(
               messages=[{"role": "system", "content": home_prompt}],
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_tokens=100
           )
           treatment_parts.append("\nHome Care:\n" + home_completion.choices[0].message.content.strip())
       
       treatment = "\n".join(treatment_parts)
       
       citations = []
       citations.extend(diagnosis_docs)
       citations.extend(treatment_docs)
       
       return {
           "status": "success",
           "diagnosis": diagnosis,
           "treatment": treatment,
           "citations": citations
       }
           
   except Exception as e:
       try:
           minimal_prompt = f"List possible diagnoses for: {initial_complaint}"
           fallback_completion = groq_client.chat.completions.create(
               messages=[{"role": "system", "content": minimal_prompt}],
               model="llama-3.3-70b-versatile",
               temperature=0.2,
               max_tokens=50
           )
           return {
               "status": "fallback",
               "diagnosis": fallback_completion.choices[0].message.content.strip(),
               "treatment": "Please consult a healthcare provider for specific treatment recommendations.",
               "citations": []
           }
       except Exception as e2:
           return {
               "status": "error",
               "message": f"Error in diagnosis/treatment: {str(e)}. Fallback also failed: {str(e2)}",
               "diagnosis": "Error generating diagnosis",
               "treatment": "Error generating treatment",
               "citations": []
           }
def store_question(question_data: dict) -> None:
    """Store a question in the global array."""
    global structured_questions_array
    structured_questions_array.append(question_data)

def parse_question_data(question: str, options: list, answer: str, matrix_output: dict, citations: list) -> dict:
    """Parse a single question's data into the structured format."""
    global structured_questions_array
    
    formatted_options = [f"{i+1}: {opt['text']}" for i, opt in enumerate(options)]
    selected_idx = next((i+1 for i, opt in enumerate(options) if opt['text'] == answer), None)
    sources = [f"{cite['source']} (relevance: {cite['score']:.2f})" for cite in citations] if citations else []
    
    question_data = {
        "question": question,
        "options": formatted_options,
        "selected_option": selected_idx,
        "pattern": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "confidence": matrix_output.get('confidence', 0.5),
        "selected_agent": matrix_output.get('selected_agent', 'optimist'),
        "weights": [
            matrix_output.get('weights', {}).get('optimist', 0.5),
            matrix_output.get('weights', {}).get('pessimist', 0.5)
        ],
        "sources": sources
    }
    
    structured_questions_array.append(question_data)
    return question_data

def parse_examination_text(examination_text: str) -> tuple[str, list[str]]:
    """Parse examination text using "#:" delimiter."""
    parts = examination_text.split("#:")
    
    if len(parts) < 2:
        raise ValueError("Invalid examination format - missing '#:' delimiter")
        
    examination = parts[0].strip()
    options = [opt.strip() for opt in parts[1:] if opt.strip()]
    
    return examination, options

def store_examination(examination_text: str, selected_option: int) -> Dict[str, Any]:
    """Store examination data in the examination history."""
    global examination_history
    
    try:
        examination, options = parse_examination_text(examination_text)
        
        examination_entry = {
            "examination": examination,
            "options": options,
            "selected_option": selected_option
        }
        
        examination_history.append(examination_entry)
        return {"status": "success", "data": examination_entry}
        
    except Exception as e:
        return {"status": "error", "message": f"Error storing examination: {str(e)}"}

def get_assessment_data() -> Dict[str, Any]:
    """Get current state of assessment data."""
    return {
        "questions": structured_questions_array,
        "examinations": examination_history
    }

def initialize_assessment() -> None:
    """Initialize/reset assessment arrays."""
    global structured_questions_array, examination_history
    structured_questions_array = []
    examination_history = []

def main(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main assessment flow, now taking responses as input instead of CLI interaction."""
    try:
        # Initialize arrays
        initialize_assessment()
        
        # Process initial responses
        initial_responses = get_initial_responses(responses)
        
        # Get follow-up responses
        followup_state = get_followup_questions(initial_responses)
        followup_responses = followup_state["responses"]
        
        # Get examination responses
        exam_state = get_followup_exams(initial_responses, followup_responses)
        exam_responses = exam_state["responses"]

        # Get diagnosis and treatment
        results = get_diagnosis_and_treatment(
            initial_responses,
            followup_responses,
            exam_responses
        )
        
        # Return comprehensive results
        return {
            "status": "success",
            "initial_responses": initial_responses,
            "followup_responses": followup_responses,
            "exam_responses": exam_responses,
            "diagnosis": results.get("diagnosis"),
            "treatment": results.get("treatment"),
            "citations": results.get("citations", []),
            "assessment_data": get_assessment_data()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
