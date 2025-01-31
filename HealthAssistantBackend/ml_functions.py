import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
import openai
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")


"""
Questions.  May be follow up question.  Till done. Exams. May be follow up questions or exams. Till done.  Diagnosis.  Treatment.  
Sometimes follow-up. Usually when to call a doctor.  Usually prevention. 
"""

class ChatStage:
    """Enum-like class for chat stages"""
    INITIAL_QUESTIONS = 'initial_questions'
    FOLLOWUP = 'followup'
    EXAM_RECOMMENDATIONS = 'exam_recommendations'
    EXAM_SELECTION = 'exam_selection'
    EXAM_RESULTS = 'exam_results'
    DIAGNOSIS = 'diagnosis'
    TREATMENT_PLAN = 'treatment_plan'
    COMPLETE = 'complete'

class ValidationResult:
    """Class to hold question validation results"""
    def __init__(self, is_valid: bool, score: float, reason: str = None):
        self.is_valid = is_valid
        self.score = score
        self.reason = reason

def getIndex():
    """Get Pinecone index."""
    return pc.Index("final-asha")

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

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        return float(cosine_similarity([embedding1], [embedding2])[0][0])
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def generate_recommended_exams(initial_complaint: str, conversation_history: list) -> List[Dict[str, Any]]:
    """Generate list of recommended medical exams based on symptoms."""
    try:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        prompt = f'''Based on:
        Initial complaint: "{initial_complaint}"
        Conversation history: {history_text}
        
        Generate a list of recommended medical exams/tests appropriate for these symptoms.
        Consider:
        1. Standard diagnostic protocols
        2. Symptom presentation
        3. Risk factors
        4. Cost-effectiveness
        
        Return a JSON array of recommended exams with format:
        {{"exams": [
            {{"name": "exam name",
              "description": "brief description",
              "rationale": "why needed",
              "priority": "urgent/high/medium/low"}}
        ]}}'''
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3
        )
        
        return eval(response.choices[0].message.content.strip())['exams']
    except Exception as e:
        print(f"Error generating exam recommendations: {e}")
        return []

def judge_question(question: str, previous_questions: List[str], initial_complaint: str, current_stage: str) -> Dict[str, Any]:
    """Judge if a question is appropriate and determine next actions with enhanced completion criteria."""
    try:
        questions_context = "\n".join([f"- {q}" for q in previous_questions])
        
        prompt = f'''Evaluate this medical assessment question and determine if enough information has been gathered.
        Initial complaint: "{initial_complaint}"
        Current stage: {current_stage}
        Previous questions and responses:
        {questions_context}
        
        New question: "{question}"
        
        COMPLETION CRITERIA:
        1. Primary Symptoms (REQUIRED):
           - Duration and onset timing
           - Severity and progression pattern
           - Location and character of symptoms
           - Aggravating/alleviating factors
        
        2. Associated Factors (REQUIRED):
           - Related symptoms
           - Previous episodes
           - Impact on daily activities
           - Current medications/treatments tried
        
        3. Medical Context (REQUIRED):
           - Relevant medical history
           - Family history if applicable
           - Current medications
           - Allergies
        
        4. Risk Assessment (REQUIRED):
           - Red flag symptoms checked
           - Severity indicators evaluated
           - Impact on daily life assessed
           
        5. Specific Conditions (If Applicable):
           - Chronic disease management
           - Mental health assessment
           - Physical limitations
           - Social support system
        
        EARLY COMPLETION TRIGGERS:
        1. Severe Symptoms:
           - Acute severe pain
           - Difficulty breathing
           - Altered consciousness
           - Severe bleeding
           
        2. Clear Emergency Indicators:
           - Chest pain with risk factors
           - Stroke symptoms
           - Severe trauma
           - Acute onset of severe symptoms
        
        3. Obvious Diagnosis Path:
           - Classic presentation of condition
           - Clear symptom pattern established
           - Diagnostic criteria met
        
        Analyze and return a JSON object with:
        {{"approved": boolean,  # Is this question appropriate?
          "reason": string,    # Explanation for decision
          "suggested_replacement": string or null,  # Better question if needed
          "assessment_complete": boolean,  # Are completion criteria met?
          "completion_details": {
            "primary_symptoms_complete": boolean,
            "associated_factors_complete": boolean,
            "medical_context_complete": boolean,
            "risk_assessment_complete": boolean,
            "missing_elements": [string]
          },
          "next_stage": string,
          "emergency_referral": boolean,  # Should patient be referred for emergency care?
          "confidence_level": float  # 0-1 scale of confidence in completion assessment
        }}'''

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1
        )
        
        result = eval(response.choices[0].message.content.strip())
        
        # Add logic for emergency situations
        if result.get("emergency_referral", False):
            return {
                "approved": False,
                "reason": "EMERGENCY REFERRAL NEEDED - " + result.get("reason", ""),
                "suggested_replacement": None,
                "assessment_complete": True,
                "next_stage": ChatStage.TREATMENT_PLAN,
                "emergency": True
            }
            
        # Only mark assessment complete if confidence is high enough
        assessment_complete = (
            result.get("assessment_complete", False) and 
            result.get("confidence_level", 0) > 0.8
        )
        
        return {
            "approved": result.get("approved", False),
            "reason": result.get("reason", "Unknown"),
            "suggested_replacement": result.get("suggested_replacement"),
            "assessment_complete": assessment_complete,
            "completion_details": result.get("completion_details", {}),
            "next_stage": result.get("next_stage", current_stage),
            "emergency": False
        }
        
    except Exception as e:
        print(f"Error in judge_question: {e}")
        return {
            "approved": True,
            "reason": "Error in judgment process",
            "suggested_replacement": None,
            "assessment_complete": False,
            "next_stage": current_stage,
            "emergency": False
        }

def validate_question_relevance(question: str, context: str, initial_complaint: str) -> ValidationResult:
    """Validate if a question is relevant to the medical context and complaint."""
    try:
        question_embedding = get_embedding(question)
        context_embedding = get_embedding(context)
        complaint_embedding = get_embedding(initial_complaint)
        
        context_similarity = calculate_similarity(question_embedding, context_embedding)
        complaint_similarity = calculate_similarity(question_embedding, complaint_embedding)
        
        prompt = f'''Evaluate if this follow-up question is relevant and appropriate:
        Initial complaint: "{initial_complaint}"
        Medical context: "{context}"
        Question: "{question}"
        
        Rules:
        1. Question must be directly related to the complaint or its symptoms
        2. Question must be medically relevant
        3. Question must be appropriate for the current context
        4. Question must be focused and specific
        
        Return ONLY:
        - "valid" if the question is appropriate and relevant
        - "invalid: [reason]" if not appropriate
        '''
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1
        )
        
        validation = response.choices[0].message.content.strip().lower()
        is_valid = validation.startswith("valid")
        reason = validation.split(": ")[1] if not is_valid else None
        
        score = (context_similarity + complaint_similarity) / 2
        
        return ValidationResult(
            is_valid=is_valid and score > 0.3,
            score=score,
            reason=reason
        )
    except Exception as e:
        print(f"Error validating question: {e}")
        return ValidationResult(is_valid=True, score=1.0)

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

def generate_diagnosis(context: dict, exam_results: str) -> Dict[str, Any]:
    """Generate diagnosis based on symptoms and exam results."""
    try:
        conversation_history = context.get('conversationHistory', [])
        initial_complaint = context.get('screeningAnswers', [{}])[-1].get('answer', '')
        
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        prompt = f'''Based on:
        Initial complaint: "{initial_complaint}"
        Conversation history: {history_text}
        Exam results: {exam_results}
        
        Generate a comprehensive diagnosis. Include:
        1. Primary diagnosis
        2. Confidence level
        3. Supporting findings
        4. Differential diagnoses
        5. Risk factors
        
        Return a JSON object with format:
        {{"diagnosis": string,
          "confidence": "high/medium/low",
          "findings": [string],
          "differentials": [string],
          "risk_factors": [string]}}'''
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3
        )
        
        return eval(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating diagnosis: {e}")
        return {}

def generate_treatment_plan(diagnosis: Dict[str, Any], context: dict) -> Dict[str, Any]:
    """Generate treatment plan based on diagnosis."""
    try:
        initial_complaint = context.get('screeningAnswers', [{}])[-1].get('answer', '')
        
        prompt = f'''Based on:
        Initial complaint: "{initial_complaint}"
        Diagnosis: {diagnosis.get('diagnosis', 'Unknown')}
        Findings: {diagnosis.get('findings', [])}
        
        Generate a comprehensive treatment plan including:
        1. Immediate actions/interventions
        2. Medications (if needed)
        3. Lifestyle modifications
        4. Follow-up care
        5. Monitoring requirements
        6. Warning signs
        7. Prevention strategies
        
        Return a JSON object with format:
        {{"immediate_actions": [string],
          "medications": [string],
          "lifestyle": [string],
          "followup": [string],
          "monitoring": [string],
          "warnings": [string],
          "prevention": [string]}}'''
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3
        )
        
        return eval(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating treatment plan: {e}")
        return {}

def generate_fallback_question(initial_complaint: str, stage: str) -> Dict[str, Any]:
    """Generate a fallback question when normal question generation fails."""
    if stage == ChatStage.INITIAL_QUESTIONS:
        return {
            "question": "How long have you been experiencing these symptoms?",
            "type": "MC",
            "options": [
                {"id": 1, "text": "Less than 24 hours"},
                {"id": 2, "text": "1-3 days"},
                {"id": 3, "text": "4-7 days"},
                {"id": 4, "text": "More than a week"},
                {"id": "other", "text": "Other (please specify)"}
            ]
        }
    elif stage == ChatStage.FOLLOWUP:
        return {
            "question": "How severe would you rate your symptoms?",
            "type": "MC",
            "options": [
                {"id": 1, "text": "Mild - noticeable but not disruptive"},
                {"id": 2, "text": "Moderate - somewhat disruptive"},
                {"id": 3, "text": "Severe - significantly disruptive"},
                {"id": 4, "text": "Very severe - cannot perform daily activities"},
                {"id": "other", "text": "Other (please specify)"}
            ]
        }
    else:
        return {
            "question": "Would you like to proceed with the assessment?",
            "type": "MC",
            "options": [
                {"id": "complete", "text": "Complete Assessment"},
                {"id": "continue", "text": "Ask Another Question"},
                {"id": "restart", "text": "Start Over"}
            ]
        }

def process_chat_response(message: str, context: dict, stage: str = ChatStage.INITIAL_QUESTIONS) -> Dict[str, Any]:
    """Process chat messages and generate appropriate responses."""
    try:
        print(f"Processing message in stage: {stage}")
        conversation_history = context.get('conversationHistory', [])
        previous_questions = [msg['content'] for msg in conversation_history if msg['role'] == 'bot']
        
        # Handle "Continue to Exam Recommendations" selection
        if message.lower() == "continue to exam recommendations":
            try:
                recommended_exams = generate_recommended_exams(initial_complaint, conversation_history)
                
                if not recommended_exams:
                    raise Exception("No exam recommendations generated")
                
                # Group exams by priority
                exams_by_priority = {}
                for exam in recommended_exams:
                    priority = exam['priority']
                    if priority not in exams_by_priority:
                        exams_by_priority[priority] = []
                    exams_by_priority[priority].append(exam)
                
                # Format recommendations with grouping
                formatted_sections = []
                for priority in ['urgent', 'high', 'medium', 'low']:
                    if priority in exams_by_priority:
                        formatted_sections.append(f"\n{priority.upper()} PRIORITY:")
                        for exam in exams_by_priority[priority]:
                            formatted_sections.append(
                                f"\n• {exam['name']}\n"
                                f"  Description: {exam['description']}\n"
                                f"  Rationale: {exam['rationale']}\n"
                                f"  Preparation: {exam['preparation']}"
                            )
                
                return {
                    "question": "Based on your symptoms, these medical exams are recommended:" + 
                               "\n\n" + "\n".join(formatted_sections) + 
                               "\n\nHow would you like to proceed?",
                    "type": "MC",
                    "options": [
                        {"id": "proceed", "text": "Schedule recommended exams"},
                        {"id": "enter_results", "text": "Enter exam results (if already completed)"},
                        {"id": "modify", "text": "Request different exams"},
                        {"id": "restart", "text": "Start Over"}
                    ],
                    "stage": ChatStage.EXAM_RECOMMENDATIONS,
                    "recommended_exams": recommended_exams
                }
            except Exception as e:
                print(f"Error in exam recommendations: {e}")
                return {
                    "question": "I apologize, but I encountered an error generating exam recommendations. Would you like to:",
                    "type": "MC",
                    "options": [
                        {"id": "retry", "text": "Try again"},
                        {"id": "continue", "text": "Continue without exams"},
                        {"id": "restart", "text": "Start over"}
                    ],
                    "stage": current_stage
                }

        # Get initial complaint
        initial_complaint = ""
        if context.get('screeningAnswers'):
            initial_complaint = context['screeningAnswers'][-1].get('answer', '')

        # Handle specific commands/stages
        if message.lower() == "view assessment results":
            # Generate exam recommendations
            recommended_exams = generate_recommended_exams(initial_complaint, conversation_history)
            formatted_exams = [
                f"{exam['name']} ({exam['priority']} priority)\n{exam['description']}\nRationale: {exam['rationale']}"
                for exam in recommended_exams
            ]
            
            return {
                "question": "Based on your symptoms, these medical exams are recommended:\n\n" + 
                           "\n\n".join(formatted_exams) + "\n\nWould you like to proceed with these exams?",
                "type": "MC",
                "options": [
                    {"id": "proceed", "text": "Proceed with recommended exams"},
                    {"id": "modify", "text": "Request different exams"},
                    {"id": "skip", "text": "Skip to diagnosis"},
                    {"id": "restart", "text": "Start Over"}
                ],
                "stage": ChatStage.EXAM_SELECTION,
                "recommended_exams": recommended_exams
            }
            
        elif message.lower() == "proceed with recommended exams":
            return {
                "question": "Please enter the results of the completed exams.",
                "type": "FREE",
                "stage": ChatStage.EXAM_RESULTS
            }
            
        elif stage == ChatStage.EXAM_RESULTS:
            # Process exam results and generate diagnosis
            diagnosis = generate_diagnosis(context, message)
            context['diagnosis'] = diagnosis
            
            return {
                "question": f"Based on the exam results, here's the preliminary diagnosis:\n\n" +
                           f"Diagnosis: {diagnosis.get('diagnosis', 'Unknown')}\n" +
                           f"Confidence: {diagnosis.get('confidence', 'Unknown')}\n\n" +
                           "Would you like to proceed with generating a treatment plan?",
                "type": "MC",
                "options": [
                    {"id": "treatment", "text": "View Treatment Plan"},
                    {"id": "details", "text": "View Detailed Diagnosis"},
                    {"id": "restart", "text": "Start Over"}
                ],
                "stage": ChatStage.TREATMENT_PLAN
            }
            
        elif message.lower() == "view treatment plan":
            # Generate treatment plan
            treatment_plan = generate_treatment_plan(context.get('diagnosis', {}), context)
            context['treatment_plan'] = treatment_plan
            
            return {
                "question": "Treatment Plan:\n\n" +
                           "Immediate Actions:\n" + "\n".join(treatment_plan.get('immediate_actions', [])) + "\n\n" +
                           "Medications:\n" + "\n".join(treatment_plan.get('medications', [])) + "\n\n" +
                           "Lifestyle Changes:\n" + "\n".join(treatment_plan.get('lifestyle', [])) + "\n\n" +
                           "Follow-up:\n" + "\n".join(treatment_plan.get('followup', [])) + "\n\n" +
                           "Warning Signs:\n" + "\n".join(treatment_plan.get('warnings', [])),
                "type": "MC",
                "options": [
                    {"id": "download", "text": "Download Treatment Plan"},
                    {"id": "restart", "text": "Start New Assessment"}
                ],
                "stage": ChatStage.COMPLETE
            }

        # Don't check message count first - let the judge decide

        try:
            # Get embeddings and search vector DB
            embedding = get_embedding(message)
            index = getIndex()
            relevant_docs = vectorQuotes(embedding, index)
            
            if not relevant_docs:
                return generate_fallback_question(initial_complaint, stage)
                
            combined_context = " ".join([doc["text"] for doc in relevant_docs])
            
            # Generate question
            prompt = f'''Based on:
            - Initial complaint: "{initial_complaint}"
            - Current message: "{message}"
            - Medical context: {combined_context}
            - Stage: {stage}
            
            Generate ONE focused, relevant follow-up question.
            Follow standard medical assessment order:
            1. Duration and onset
            2. Characteristics and severity
            3. Associated symptoms
            4. Aggravating/relieving factors
            5. Impact on daily life
            
            Return only the question text.'''
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3
            )
            
            question = response.choices[0].message.content.strip()
            
            # Submit to judge LLM
            judgment = judge_question(question, previous_questions, initial_complaint, stage)
            
            if not judgment['approved']:
                if judgment['suggested_replacement']:
                    question = judgment['suggested_replacement']
                else:
                    return generate_fallback_question(initial_complaint, stage)
            
            if judgment['assessment_complete']:
                # Move to exam recommendations
                recommended_exams = generate_recommended_exams(initial_complaint, conversation_history)
                formatted_exams = [
                    f"{exam['name']} ({exam['priority']} priority)\n{exam['description']}\nRationale: {exam['rationale']}"
                    for exam in recommended_exams
                ]
                
                return {
                    "question": "Initial assessment complete. Based on your symptoms, these medical exams are recommended:\n\n" + 
                               "\n\n".join(formatted_exams) + "\n\nWould you like to proceed?",
                    "type": "MC",
                    "options": [
                        {"id": "proceed", "text": "Proceed with recommended exams"},
                        {"id": "modify", "text": "Request different exams"},
                        {"id": "skip", "text": "Skip to diagnosis"},
                        {"id": "restart", "text": "Start Over"}
                    ],
                    "stage": ChatStage.EXAM_RECOMMENDATIONS,
                    "recommended_exams": recommended_exams
                }
            
            # Generate options for the question
            options_prompt = f'''Generate 4-5 possible answers for: "{question}"
            Requirements:
            - Clear, concise options
            - Mutually exclusive
            - Cover likely scenarios
            - Include severity levels if applicable
            Return each option on a new line.'''
            
            options_response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "system", "content": options_prompt}],
                temperature=0.2
            )
            
            # Parse options
            options = []
            for i, opt in enumerate(options_response.choices[0].message.content.strip().split('\n')):
                if opt.strip():
                    options.append({"id": i+1, "text": opt.strip()})
            
            # Add "Other" option
            options.append({"id": "other", "text": "Other (please specify)"})
            
            # Add Continue option after 3 questions
            if len(conversation_history) >= 3:
                options.insert(-1, {"id": "continue", "text": "Continue to Exam Recommendations"})
            
            return {
                "question": question,
                "type": "MC",
                "options": options,
                "stage": judgment['next_stage'],
                "remaining_questions": 20 - len(conversation_history)
            }
            
        except Exception as inner_e:
            print(f"Error generating question: {inner_e}")
            return generate_fallback_question(initial_complaint, stage)
            
    except Exception as e:
        print(f"Error in process_chat_response: {e}")
        return {
            "question": "I apologize, but I encountered an error. Please choose:",
            "type": "MC",
            "options": [
                {"id": 1, "text": "Try again"},
                {"id": 2, "text": "Start over"},
                {"id": "complete", "text": "Complete Assessment"}
            ],
            "stage": stage
        }

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
        "question": "Please describe what brings you here today",
        "type": "FREE"
    }
]