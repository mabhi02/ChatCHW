from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_completion(prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
    """
    Get completion from OpenAI's GPT-4 API.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",  # Using GPT-4-mini
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




"""
Medical Assessment System Prompts
=================================
This file contains all the prompts used in the medical assessment system.
"""

# --- SPECIAL EXAM AND TREATMENT BUTTON TEMPLATES ---
RDT_EXAM_PROMPT = '''EXAM: Give the RDT (Rapid Diagnostic Test for Malaria).
Button0: TELL ME HOW
Button1: NEGATIVE for malaria
Button2: POSITIVE for malaria
Button3: RDT not given, no supplies
Button4: RDT not given, other _______ [microphone]
Button5: Other __________ [microphone]'''

ORS_ZINC_PROMPT = '''Screen 1: Give ORS, Zinc and go to clinic.
Screen 2: Explains ORS. Asks if ORS given.
CARE: Give 2 sachets of ORS to the caregiver
Button1: ORS given
Button2: ORS not given, no supplies
Button3: ORS not given, other _______ [microphone]
Button4: Other __________ [microphone]

Screen 3: Explains zinc. Asks if zinc given.
Screen 4: Refer to clinic
CARE: Refer to clinic
Button1: Gave referral
Button2: Did not give referral. Why? _____ [microphone]
Button3: Other __________ [microphone]'''

FAINTING_PROMPT = 'This case is outside my scope. Go to the clinic.'

class MedicalPrompts:
    """Collection of all prompts used in the medical assessment system."""
    
    # Diagnosis and Treatment Prompts
    DIAGNOSIS_PROMPT = """Patient information:
Initial complaint: {initial_complaint}
Key findings: {key_findings}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{diagnosis_content}

IMPORTANT INSTRUCTIONS:
1. The medical guide is ALWAYS correct and should be your primary source of information. Always directly reference and draw from the guide's content as your main justification.
2. Base your response primarily on the medical guide information above, but you may make reasonable inferences based on:
   - EXTREMELY limited resources available in third-world settings (e.g., no advanced imaging, limited medications, basic equipment only)
   - Common medical practices in resource-constrained environments where access to healthcare is limited
   - Basic medical principles that would be known to community health workers in rural/remote areas
3. Clearly distinguish between:
   - Direct recommendations from the medical guide (these take precedence)
   - Inferences you are making based on the guide and context
   - Adaptations necessary for third-world resource constraints
4. If the medical guide doesn't contain sufficient information for a specific aspect, state this clearly.
5. Format your response to be helpful to a community health worker in a third-world setting, considering severe resource limitations.
6. Use terminology appropriate for community health workers in low-resource, third-world settings.
7. If you cannot make a confident recommendation with the given information, state this clearly and recommend referral to the nearest available facility.
8. Do not use any patient names or identifiers in your response.
9. Always consider:
   - Limited access to transportation for referrals
   - Limited availability of medications and supplies
   - Basic equipment and facilities only
   - Cultural and language barriers
   - Limited access to follow-up care

What likely conditions might explain the patient's symptoms, based on the medical guide information and reasonable inferences for a third-world setting?"""

    IMMEDIATE_CARE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{immediate_content}

IMPORTANT INSTRUCTIONS:
1. The medical guide is ALWAYS correct and should be your primary source of information. Always directly reference and draw from the guide's content as your main justification.
2. Base your response primarily on the medical guide information above, but you may make reasonable inferences based on:
   - EXTREMELY limited resources available in third-world settings (e.g., no advanced equipment, limited medications, basic supplies only)
   - Common medical practices in resource-constrained environments where access to healthcare is limited
   - Basic medical principles that would be known to community health workers in rural/remote areas
3. Clearly distinguish between:
   - Direct recommendations from the medical guide (these take precedence)
   - Inferences you are making based on the guide and context
   - Adaptations necessary for third-world resource constraints
4. If the medical guide doesn't contain sufficient information for immediate care, state this clearly.
5. Present immediate care steps considering severe resource limitations in third-world settings.
6. Use terminology appropriate for community health workers in low-resource settings.
7. If you cannot make a confident recommendation with the given information, state this clearly and recommend immediate referral to the nearest available facility.
8. Do not use any patient names or identifiers in your response.
9. Always consider:
   - Limited access to transportation for emergencies
   - Limited availability of medications and supplies
   - Basic equipment and facilities only
   - Cultural and language barriers
   - Limited access to follow-up care

Based on the medical guide information and reasonable inferences for a third-world setting, what immediate care steps should be taken?"""

    HOME_CARE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{home_content}

IMPORTANT INSTRUCTIONS:
1. The medical guide is ALWAYS correct and should be your primary source of information. Always directly reference and draw from the guide's content as your main justification.
2. Base your response primarily on the medical guide information above, but you may make reasonable inferences based on:
   - EXTREMELY limited resources available in third-world settings (e.g., limited access to clean water, basic hygiene supplies, traditional remedies)
   - Common home care practices in resource-constrained environments where access to healthcare is limited
   - Basic medical principles that would be known to community health workers in rural/remote areas
3. Clearly distinguish between:
   - Direct recommendations from the medical guide (these take precedence)
   - Inferences you are making based on the guide and context
   - Adaptations necessary for third-world resource constraints
4. If the medical guide doesn't contain specific home care information, state this clearly.
5. Present home care instructions considering severe resource limitations in third-world settings.
6. Use terminology appropriate for community health workers in low-resource settings.
7. If you cannot make a confident recommendation with the given information, state this clearly and recommend referral to the nearest available facility.
8. Do not use any patient names or identifiers in your response.
9. Always consider:
   - Limited access to clean water and sanitation
   - Limited availability of medications and supplies
   - Cultural practices and traditional remedies
   - Language and literacy barriers
   - Limited access to follow-up care

Based on the medical guide information and reasonable inferences for a third-world setting, what home care instructions should be provided?"""

    REFERRAL_GUIDANCE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{referral_content}

IMPORTANT INSTRUCTIONS:
1. The medical guide is ALWAYS correct and should be your primary source of information. Always directly reference and draw from the guide's content as your main justification.
2. Base your response primarily on the medical guide information above, but you may make reasonable inferences based on:
   - EXTREMELY limited resources available in third-world settings (e.g., limited transportation, distant facilities, basic care only)
   - Common referral practices in resource-constrained environments where access to healthcare is limited
   - Basic medical principles that would be known to community health workers in rural/remote areas
3. Clearly distinguish between:
   - Direct recommendations from the medical guide (these take precedence)
   - Inferences you are making based on the guide and context
   - Adaptations necessary for third-world resource constraints
4. If the medical guide doesn't contain specific referral criteria, state this clearly.
5. Present referral guidance considering severe resource limitations in third-world settings.
6. Use terminology appropriate for community health workers in low-resource settings.
7. If you cannot make a confident recommendation with the given information, state this clearly and recommend conservative referral to the nearest available facility.
8. Do not use any patient names or identifiers in your response.
9. Always consider:
   - Limited access to transportation
   - Distance to nearest facilities
   - Limited availability of specialized care
   - Cultural and language barriers
   - Cost implications for the patient

Based on the medical guide information and reasonable inferences for a third-world setting, when should this patient be referred to a higher level facility?"""

    # Follow-up Question Generation Prompts
    FOLLOWUP_QUESTION_PROMPT = """Based on the patient's initial complaint: "{initial_complaint}"

Previous questions asked:
{previous_questions}

Relevant medical context:
{combined_context}

IMPORTANT INSTRUCTIONS:
1. The medical guide is ALWAYS correct and should be your primary source of information. Always directly reference and draw from the guide's content as your main justification.
2. Base your question on the medical guide information above, but you may make reasonable inferences based on:
   - EXTREMELY limited resources available in third-world settings
   - Common assessment practices in resource-constrained environments
   - Basic medical principles that would be known to community health workers in rural/remote areas
3. Generate ONE focused, relevant follow-up question that is different from the previous questions.
4. Do not ask compound questions (e.g., "Do you have fever and cough?").
5. Focus on one specific metric or symptom at a time.
6. Consider severe resource limitations in third-world settings when formulating questions.
7. Do not use any patient names or identifiers in your question.
8. Always consider:
   - Limited access to healthcare
   - Cultural and language barriers
   - Basic understanding of medical concepts
   - Available resources for treatment

Return only the question text."""

    QUESTION_OPTIONS_PROMPT = """Generate 4 concise answers for: "{question}"
Clear, mutually exclusive options.
Return each option on a new line (1-4)."""

    # Examination Prompts
    EXAMINATION_RECOMMENDATION_PROMPT = """Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {key_symptoms}

Previous exams: {previous_exams}

IMPORTANT INSTRUCTIONS:
1. If the case is about malaria or fever in a malaria area, you MUST provide the actual step-by-step procedure for performing a rapid diagnostic test (RDT) for malaria. Use all available information from the guide. If the guide does not provide explicit steps, synthesize them using safe, practical steps that could be performed in a third-world/low-resource setting (you may use your pretrained knowledge ONLY for the procedure, not for diagnosis or treatment). Do NOT default to generic findings or referral if a real test is present and can be described.
2. For all other cases, you MUST synthesize and conduct a concrete, step-by-step examination using all relevant information from the medical guide, even if the guide does not present it as a formal procedure. If the guide provides the necessary context (such as how to count breaths, what constitutes fast breathing, or what danger signs to look for), you must construct the examination and findings based on that information.
3. The possible findings must be actionable, clinically relevant, and, where appropriate, presented as ranges or categories that provide valuable information for decision-making (e.g., "RDT positive for malaria", "RDT negative for malaria", "RDT not performed: no supplies", "Other: specify").
4. NEVER default to generic findings or referral if a reasonable, context-based exam can be constructed from the guide or synthesized from safe, practical steps.
5. Do not include any first-world exams (like MRI, CT, Colonoscopy, etc.).
6. If the guide mentions a test or examination but doesn't provide steps, you MUST synthesize the steps from the context and related sections in the guide, or use safe, practical steps from your pretrained knowledge for the procedure only.

Recommend ONE essential examination in this format:
[Examination name]
[Detailed step-by-step procedure to perform the examination. Each step should be numbered and clear enough for a medical professional to follow exactly. If the guide provides specific steps, use those exact steps. If not, synthesize them from the context or use safe, practical steps from your pretrained knowledge for the procedure only.]
#:[First possible finding - must be actionable, context-appropriate, and relevant to the exam]
#:[Second possible finding - must be actionable, context-appropriate, and relevant to the exam]
#:[Third possible finding - must be actionable, context-appropriate, and relevant to the exam]
#:[Fourth possible finding - must be actionable, context-appropriate, and relevant to the exam]

If the medical guide truly does not provide enough information to construct any examination, respond with:
"The medical guide does not provide specific examination procedures or enough information to construct an examination for this condition."

If the medical guide provides partial information but no clear procedure, respond with:
"While the medical guide does not provide a formal examination procedure, here is some relevant information that may be helpful:"
[Include any relevant assessment guidance from the guide]"""

    # Fallback Prompts
    MINIMAL_DIAGNOSIS_PROMPT = """IMPORTANT: You can ONLY use information from the WHO medical guide for community health workers for this response.

For a patient with '{initial_complaint}', what does the guide recommend?

If the medical guide doesn't contain relevant information about this condition, simply state that the guide doesn't provide specific information for this complaint and the patient should be referred to a health facility.

DO NOT use any medical knowledge from your pretraining."""

    # System Configuration Prompts
    SYSTEM_CONTEXT_PROMPT = """You are a medical assessment assistant helping community health workers provide care based strictly on WHO medical guidelines. Always prioritize patient safety and follow the medical guide recommendations exactly as written."""

    # Embedding Query Templates
    EMBEDDING_QUERIES = {
        'diagnosis': "{complaint} diagnosis",
        'immediate_care': "{complaint} immediate care steps",
        'home_care': "{complaint} home care advice", 
        'referral': "{complaint} when to refer"
    }

    # Error Messages
    ERROR_MESSAGES = {
        'no_diagnosis_docs': "Unable to provide diagnosis. No relevant information found in the medical guide. Please refer the patient to the nearest health facility.",
        'no_treatment_docs': "Unable to provide treatment recommendations. No relevant information found in the medical guide. Please refer the patient to the nearest health facility.",
        'retrieval_error': "Error retrieving information from the medical guide. Please refer the patient to a health facility for proper evaluation.",
        'fallback_treatment': "The medical guide doesn't provide specific treatment information for this condition. Please refer the patient to a health facility for proper evaluation and treatment."
    }

    # Default Examination Findings
    DEFAULT_EXAMINATION_FINDINGS = [
        "No specific finding available - refer to higher facility",
        "Unable to perform examination based on medical guide", 
        "Need additional clinical assessment",
        "Consider general observation only"
    ]

    PARTIAL_INFO_EXAMINATION_FINDINGS = [
        "Consider referring to medical professional for proper examination",
        "Use this information as supplementary guidance only",
        "Document observations based on general assessment", 
        "Consult with supervisor about next steps"
    ]

    NORMAL_EXAMINATION_FINDINGS = [
        "Normal finding",
        "Abnormal finding requiring further assessment",
        "Inconclusive finding - may need additional tests",
        "Unable to determine based on current examination"
    ]

    @classmethod
    def get_diagnosis_prompt(cls, initial_complaint: str, key_findings: str, diagnosis_content: str) -> str:
        """Get the formatted diagnosis prompt."""
        return (
            'IMPORTANT: Your response must ONLY be these 3 bullet points, each 2-3 sentences. Do not include any other text, headers, or explanations.'
            '\n- Primary Diagnosis:'
            '\n- Differential:'
            '\n- Treatment Plan:'
            '\n\nUse only the information from the medical guide below.'
            f'\n\nPatient information:'
            f'\nInitial complaint: {initial_complaint}'
            f'\nKey findings: {key_findings}'
            f'\n\nTHE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:'
            f'\n{diagnosis_content}'
        )

    @classmethod
    def get_immediate_care_prompt(cls, immediate_content: str, initial_complaint: str) -> str:
        """Get the formatted immediate care prompt."""
        return cls.IMMEDIATE_CARE_PROMPT.format(
            immediate_content=immediate_content,
            initial_complaint=initial_complaint
        )

    @classmethod
    def get_home_care_prompt(cls, home_content: str, initial_complaint: str) -> str:
        """Get the formatted home care prompt."""
        return cls.HOME_CARE_PROMPT.format(
            home_content=home_content,
            initial_complaint=initial_complaint
        )

    @classmethod
    def get_referral_guidance_prompt(cls, referral_content: str, initial_complaint: str) -> str:
        """Get the formatted referral guidance prompt."""
        return cls.REFERRAL_GUIDANCE_PROMPT.format(
            referral_content=referral_content,
            initial_complaint=initial_complaint
        )

    @classmethod
    def get_followup_question_prompt(cls, initial_complaint: str, previous_questions: str, combined_context: str) -> str:
        """Get the formatted follow-up question prompt."""
        return cls.FOLLOWUP_QUESTION_PROMPT.format(
            initial_complaint=initial_complaint,
            previous_questions=previous_questions if previous_questions else "No previous questions yet",
            combined_context=combined_context
        )

    @classmethod
    def get_question_options_prompt(cls, question: str) -> str:
        """Get the formatted question options prompt."""
        return cls.QUESTION_OPTIONS_PROMPT.format(question=question)

    @classmethod
    def get_examination_recommendation_prompt(cls, initial_complaint: str, key_symptoms: str, previous_exams: str) -> str:
        """Get the formatted examination recommendation prompt."""
        return cls.EXAMINATION_RECOMMENDATION_PROMPT.format(
            initial_complaint=initial_complaint,
            key_symptoms=key_symptoms,
            previous_exams=previous_exams if previous_exams else "None",
            rdt_exam=RDT_EXAM_PROMPT
        )

    @classmethod
    def get_minimal_diagnosis_prompt(cls, initial_complaint: str) -> str:
        """Get the formatted minimal diagnosis prompt."""
        return cls.MINIMAL_DIAGNOSIS_PROMPT.format(initial_complaint=initial_complaint)

    @classmethod
    def get_embedding_query(cls, query_type: str, complaint: str) -> str:
        """Get the formatted embedding query."""
        template = cls.EMBEDDING_QUERIES.get(query_type, "{complaint}")
        return template.format(complaint=complaint)

    # System Messages
    SYSTEM_MESSAGES = {
        'followup_question': "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training.",
        'examination': "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training.",
        'options': "You can ONLY use information from the medical guide. Do not add any medical knowledge from your training."
    }

    # No Information Responses
    NO_INFO_RESPONSES = {
        'examination_not_available': "The medical guide does not provide specific examination procedures for this condition.",
        'proceed_options': [
            {"id": 1, "text": "Proceed with general assessment and diagnosis"},
            {"id": 2, "text": "Refer to higher level facility"},
            {"id": 3, "text": "Other (please specify)"}
        ],
        'system_error_options': [
            {"id": 1, "text": "Proceed with general assessment and diagnosis"},
            {"id": 2, "text": "Refer to higher level facility"},
            {"id": 3, "text": "Other (please specify)"}
        ]
    }

    # Assessment Summary Templates
    REFERRAL_SUMMARY_TEMPLATE = """Assessment Summary

Initial complaint: {initial_complaint}

Key findings:
{key_findings}

Diagnosis:
The medical guide does not provide specific examination procedures for this condition. 
A full diagnosis requires proper medical examination.

Recommendation:
Patient has been referred to a higher level facility for proper examination and diagnosis.

Note: This summary is based on limited information and should not be considered a complete medical assessment."""

    COMPLETE_ASSESSMENT_TEMPLATE = """Assessment Complete

- Primary Diagnosis: {primary_diagnosis}
- Differential: {differential}
- Treatment Plan: {treatment_plan}

Key References:
{citations_text}"""

    # Limitation Notes
    LIMITATION_NOTES = {
        'no_formal_exam': "\nNote: This assessment is based on limited information without formal examination procedures described in the medical guide. Consider referring for a complete medical evaluation if symptoms persist or worsen."
    }

    @classmethod
    def get_system_message(cls, message_type: str) -> str:
        """Get the appropriate system message."""
        return cls.SYSTEM_MESSAGES.get(message_type, cls.SYSTEM_CONTEXT_PROMPT)


# Prompt Templates for Different Question Types
QUESTION_TEMPLATES = {
    'pain': [
        {
            "question": "How long have you been experiencing this pain?",
            "options": [
                {"id": 1, "text": "Less than 24 hours"},
                {"id": 2, "text": "1-7 days"}, 
                {"id": 3, "text": "1-4 weeks"},
                {"id": 4, "text": "More than a month"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        },
        {
            "question": "How would you describe the severity of the pain?",
            "options": [
                {"id": 1, "text": "Mild - noticeable but not interfering with activities"},
                {"id": 2, "text": "Moderate - somewhat interfering with activities"},
                {"id": 3, "text": "Severe - significantly interfering with activities"},
                {"id": 4, "text": "Very severe - unable to perform activities"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        },
        {
            "question": "What makes the pain worse?",
            "options": [
                {"id": 1, "text": "Movement or physical activity"},
                {"id": 2, "text": "Pressure or touch"},
                {"id": 3, "text": "Specific positions"},
                {"id": 4, "text": "Nothing specific"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        }
    ],
    
    'general': [
        {
            "question": "When did your symptoms first begin?",
            "options": [
                {"id": 1, "text": "Within the last 24 hours"},
                {"id": 2, "text": "In the past week"},
                {"id": 3, "text": "Several weeks ago"},
                {"id": 4, "text": "More than a month ago"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        },
        {
            "question": "How often do you experience these symptoms?",
            "options": [
                {"id": 1, "text": "Constantly"},
                {"id": 2, "text": "Several times a day"},
                {"id": 3, "text": "A few times a week"},
                {"id": 4, "text": "Occasionally"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        },
        {
            "question": "How does this affect your daily activities?",
            "options": [
                {"id": 1, "text": "Not at all"},
                {"id": 2, "text": "Slightly limiting"},
                {"id": 3, "text": "Moderately limiting"},
                {"id": 4, "text": "Severely limiting"},
                {"id": 5, "text": "Other (please specify)"}
            ]
        }
    ]
}

# Fallback question for when all templates are exhausted
FALLBACK_QUESTION = {
    "question": "Are you experiencing any other symptoms?",
    "options": [
        {"id": 1, "text": "No other symptoms"},
        {"id": 2, "text": "Yes, mild additional symptoms"},
        {"id": 3, "text": "Yes, moderate additional symptoms"},
        {"id": 4, "text": "Yes, severe additional symptoms"},
        {"id": 5, "text": "Other (please specify)"}
    ],
    "type": "MC"
}

def get_followup_question_prompt(initial_complaint: str, known_info_text: str, previous_questions_text: str, medical_guide_content: str) -> str:
    return f'''Patient information:
Initial complaint: "{initial_complaint}"

Information we already know (DO NOT ASK ABOUT THESE AGAIN):
{known_info_text}

Previous questions already asked (DO NOT REPEAT THESE):
{previous_questions_text if previous_questions_text else "- No previous follow-up questions"}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. The question must be based directly on assessment questions mentioned in the medical guide.
4. If the medical guide mentions specific questions to ask for this condition, use those exact questions.
5. DO NOT ask about information we already know (like age or sex).
6. DO NOT repeat any previous questions that have already been asked.
7. Focus on gathering NEW information that is not already known.
8. Return only the question text without any explanation.

Based ONLY on the medical guide information, what follow-up question should be asked to assess this patient?
'''

def get_followup_options_prompt(question: str, medical_guide_content: str) -> str:
    return f'''QUESTION: "{question}"

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. Generate 4 answer options for the above question.
4. If the medical guide mentions specific answer options, use those exactly.
5. If not, create relevant options based only on the medical guide content.
6. Format as a numbered list (1-4).
7. Do not include explanations, only the options themselves.
'''

def get_examination_prompt(initial_complaint: str, symptoms_summary: str, known_info_text: str, previous_exams_text: str, medical_guide_content: str) -> str:
    return f'''Patient information:
Initial complaint: "{initial_complaint}"
Key symptoms: {symptoms_summary}

Information we already know:
{known_info_text}

Previous examinations already performed (DO NOT REPEAT THESE):
{previous_exams_text}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

CRITICAL INSTRUCTIONS:
1. You MUST provide a concrete examination procedure with detailed steps.
2. If the patient has fever, malaria symptoms, or is in a malaria area, you MUST provide the RDT (Rapid Diagnostic Test) procedure for malaria.
3. If the guide mentions ANY test, examination, or assessment procedure, you MUST provide it with detailed steps.
4. You may use your pretrained knowledge ONLY for synthesizing safe, practical examination steps if the guide provides context but not explicit steps.
5. NEVER say "the guide doesn't provide examination procedures" unless you have thoroughly searched the provided content and found absolutely nothing relevant.

REQUIRED FORMAT - FOLLOW EXACTLY:
[Examination name - be specific and clear]

[Why this examination should be conducted - explain what the guide says about this condition and why this specific exam is needed. Reference specific information from the guide.]

[Step-by-step procedure to conduct the examination:
1. [First step - be specific and actionable]
2. [Second step - be specific and actionable]
3. [Third step - be specific and actionable]
... continue with numbered steps as needed]

#:[Finding A - must be concrete and actionable]
#:[Finding B - must be concrete and actionable] 
#:[Finding C - must be concrete and actionable]
#:[Finding D - must be concrete and actionable]

SPECIAL RULES:
- For malaria/fever cases: ALWAYS provide RDT procedure with steps
- For breathing problems: ALWAYS provide respiratory rate counting procedure
- Use terminology from the guide when possible
- Make findings specific and actionable (e.g., "RDT positive", "RDT negative", "RDT not available")
- Include the reasoning from the guide about why this examination is needed
- Number each step clearly (1, 2, 3, etc.)

Based on the medical guide information above, what examination should be performed? Provide the complete procedure with findings.'''

def get_diagnosis_prompt(initial_complaint: str, symptoms_text: str, exam_results_text: str, medical_guide_content: str) -> str:
    return f'''Patient information:
Initial complaint: "{initial_complaint}"

Symptoms and history:
{symptoms_text}

Examination results:
{exam_results_text}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. Provide a diagnosis based ONLY on the medical guide information.
4. If the medical guide does not provide enough information for a diagnosis, state this clearly.
5. Format your response as:
   - Primary diagnosis (if clear from guide)
   - Differential diagnoses (if mentioned in guide)
   - Reasoning based on guide content

Based ONLY on the medical guide information, what is the diagnosis?

IMPORTANT: The final diagnosis should ONLY be these 3 bullets with 2-3 sentences each:
- Primary Diagnosis
- Differential
- Treatment plan'''

def get_treatment_prompt(initial_complaint: str, symptoms_text: str, exam_results_text: str, diagnosis: str, medical_guide_content: str) -> str:
    # Special case: fainting
    if 'faint' in initial_complaint.lower():
        return FAINTING_PROMPT
    # Special case: ORS with danger sign
    if 'ors' in initial_complaint.lower() and 'danger' in symptoms_text.lower():
        return ORS_ZINC_PROMPT
    return f'''ACTION PLAN:
Based on the diagnosis and medical guide, provide a streamlined, stepwise treatment plan with clear action items for the health worker to follow. Each action should be concise and actionable. If referral is needed, state it clearly as a final step.

Patient information:
Initial complaint: "{initial_complaint}"
Symptoms and history:
{symptoms_text}
Examination results:
{exam_results_text}
Diagnosis:
{diagnosis}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. Provide treatment recommendations as a list of action items, each on a new line, in the order they should be performed.
4. If the medical guide does not provide treatment information, state this clearly.
5. If referral is needed, include it as the last action item.
'''

def get_main_followup_question_prompt(initial_complaint: str, previous_questions: str, combined_context: str) -> str:
    return f'''Based on the patient's initial complaint: "{initial_complaint}"
                
Previous questions asked:
{previous_questions if previous_questions else "No previous questions yet"}

Relevant medical context:
{combined_context}

Generate ONE focused, relevant follow-up question that is different from the previous questions.
Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
Do not as about compound questions like "Do you have fever and cough?" or "Do you have pain in your chest or abdomen?". It should be one or the other like "Do you have fever" or "Do you have pain in your chest?".
There should be no "or" or "and" in the question as ask about one specific metric not compounded one.

Return only the question text.'''

def get_main_examination_prompt(initial_complaint: str, symptoms: str, previous_exams: str) -> str:
    return f'''Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {symptoms}

Previous exams: {previous_exams}

IMPORTANT INSTRUCTIONS:
1. ONLY recommend examinations that are explicitly mentioned in the medical guide.
2. If the guide mentions a specific examination (like RDT, blood pressure check, etc.), you MUST provide the exact procedure steps as described in the guide.
3. The possible findings must be concrete, actionable, and, where appropriate, presented as ranges or categories that provide valuable information for clinical decision-making (e.g., "Breaths per minute: less than 40", "Breaths per minute: 40–59", "Breaths per minute: 60 or more"). Do not use generic placeholders.
4. Do not include any first-world exams (like MRI, CT, Colonoscopy, etc.).
5. If the guide mentions a test or examination but doesn't provide steps, you MUST search through the guide content to find the procedure steps.
6. NEVER default to generic responses like "refer to higher facility" if the guide actually provides examination steps.

Recommend ONE essential examination in this format:
[Examination name]
[Detailed step-by-step procedure to perform the examination. Each step should be numbered and clear enough for a medical professional to follow exactly. If the guide provides specific steps, use those exact steps.]
#:[First possible finding - must be concrete, actionable, and, where appropriate, a range or category relevant to the exam]
#:[Second possible finding - must be concrete, actionable, and, where appropriate, a range or category relevant to the exam]
#:[Third possible finding - must be concrete, actionable, and, where appropriate, a range or category relevant to the exam]
#:[Fourth possible finding - must be concrete, actionable, and, where appropriate, a range or category relevant to the exam]

If the medical guide does not provide a specific examination procedure, respond with:
"The medical guide does not provide specific examination procedures for this condition."

If the medical guide provides partial information but no clear procedure, respond with:
"While the medical guide does not provide a formal examination procedure, here is some relevant information that may be helpful:"
[Include any relevant assessment guidance from the guide]''' 