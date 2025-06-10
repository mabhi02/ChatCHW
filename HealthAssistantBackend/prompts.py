"""
Medical Assessment System Prompts
=================================
This file contains all the prompts used in the medical assessment system.
"""

class MedicalPrompts:
    """Collection of all prompts used in the medical assessment system."""
    
    # Diagnosis and Treatment Prompts
    DIAGNOSIS_PROMPT = """Patient information:
Initial complaint: {initial_complaint}
Key findings: {key_findings}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{diagnosis_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain sufficient information for a specific aspect, simply say the guide doesn't provide that information.
4. Format your response to be helpful to a community health worker exactly as shown in the medical guide.
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

What likely conditions might explain the patient's symptoms, based ONLY on the medical guide information?"""

    IMMEDIATE_CARE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{immediate_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain sufficient information for immediate care, simply say the guide doesn't provide that information.
4. Present the immediate care steps exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, what immediate care steps should be taken?"""

    HOME_CARE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{home_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain specific home care information, simply say the guide doesn't provide that information.
4. Present the home care instructions exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and recommendations as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, what home care instructions should be provided?"""

    REFERRAL_GUIDANCE_PROMPT = """THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{referral_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. If the medical guide doesn't contain specific referral criteria, simply say the guide doesn't provide that information.
4. Present the referral guidance exactly as described in the medical guide for a patient with: {initial_complaint}
5. Use the same terminology and warning signs as presented in the medical guide.
6. Do not alter or simplify the medical guidance provided in the guide - present it as written.

Based ONLY on the medical guide information, when should this patient be referred to a higher level facility?"""

    # Follow-up Question Generation Prompts
    FOLLOWUP_QUESTION_PROMPT = """Based on the patient's initial complaint: "{initial_complaint}"

Previous questions asked:
{previous_questions}

Relevant medical context:
{combined_context}

Generate ONE focused, relevant follow-up question that is different from the previous questions.
Like do not ask both "How long have you had the pain?" and "How severe is the pain?", as they are too similar. It should only be like one or the other
Do not as about compound questions like "Do you have fever and cough?" or "Do you have pain in your chest or abdomen?". It should be one or the other like "Do you have fever" or "Do you have pain in your chest?".
There should be no "or" or "and" in the question as ask about one specific metric not compounded one.

Return only the question text."""

    QUESTION_OPTIONS_PROMPT = """Generate 4 concise answers for: "{question}"
Clear, mutually exclusive options.
Return each option on a new line (1-4)."""

    # Examination Prompts
    EXAMINATION_RECOMMENDATION_PROMPT = """Based on:
Initial complaint: "{initial_complaint}"
Key symptoms: {key_symptoms}

Previous exams: {previous_exams}

Recommend ONE essential examination in this format (should not be first world exams like MRI, CT, Colonoscopy, etc.):
[Examination name]
[Procedure to perform the examination. Make sure this is detailed enough for a medical professional to understand and conduct]
#:[First possible finding]
#:[Second possible finding]
#:[Third possible finding]
#:[Fourth possible finding]"""

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
        return cls.DIAGNOSIS_PROMPT.format(
            initial_complaint=initial_complaint,
            key_findings=key_findings,
            diagnosis_content=diagnosis_content
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
            previous_exams=previous_exams if previous_exams else "None"
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

    # Advanced Follow-up Question Prompts (from Flask API)
    ADVANCED_FOLLOWUP_PROMPT = """Patient information:
Initial complaint: "{initial_complaint}"

Information we already know (DO NOT ASK ABOUT THESE AGAIN):
{known_info_text}

Previous questions already asked (DO NOT REPEAT THESE):
{previous_questions_text}

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

Based ONLY on the medical guide information, what follow-up question should be asked to assess this patient?"""

    ADVANCED_OPTIONS_PROMPT = """QUESTION: "{question}"

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. Generate 4 answer options for the above question.
4. If the medical guide mentions specific answer options, use those exactly.
5. If not, create relevant options based only on the medical guide content.
6. Format as a numbered list (1-4).
7. Do not include explanations, only the options themselves."""

    # Advanced Examination Prompts (from Flask API)
    ADVANCED_EXAMINATION_PROMPT = """Patient information:
Initial complaint: "{initial_complaint}"
Key symptoms: {symptoms_summary}

Information we already know:
{known_info_text}

Previous examinations already performed (DO NOT REPEAT THESE):
{previous_exams_text}

THE FOLLOWING MEDICAL GUIDE INFORMATION IS YOUR ONLY SOURCE OF KNOWLEDGE:
{medical_guide_content}

IMPORTANT INSTRUCTIONS:
1. ONLY use information contained in the medical guide above to formulate your response.
2. Do NOT add any medical knowledge from your pretraining.
3. You MUST select one of these three formats for your response:

FORMAT 1 - WHEN THE GUIDE CLEARLY PROVIDES AN EXAMINATION PROCEDURE:
[Examination name]
[Detailed procedure steps as described in the guide]
#: [Finding 1]
#: [Finding 2]
#: [Finding 3]
#: [Finding 4]

FORMAT 2 - WHEN THE GUIDE PROVIDES NO INFORMATION ABOUT EXAMINATIONS:
"The medical guide does not provide specific examination procedures for this condition."

FORMAT 3 - WHEN THE GUIDE PROVIDES PARTIAL/INFORMAL INFORMATION BUT NO CLEAR PROCEDURE:
"While the medical guide does not provide a formal examination procedure, here is some relevant information that may be helpful:"
[Include any relevant assessment guidance from the guide]

4. Use the same terminology and procedures as presented in the medical guide.
5. DO NOT repeat any examinations that have already been performed.
6. Choose an examination that is DIFFERENT from previous ones and appropriate for the patient's symptoms.

Based ONLY on the medical guide information, what NEW examination should be performed?"""

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

Diagnosis:
{diagnosis_text}

Treatment Plan:
{treatment_text}
{limitation_note}

Key References:
{citations_text}"""

    # Limitation Notes
    LIMITATION_NOTES = {
        'no_formal_exam': "\nNote: This assessment is based on limited information without formal examination procedures described in the medical guide. Consider referring for a complete medical evaluation if symptoms persist or worsen."
    }

    @classmethod
    def get_advanced_followup_prompt(cls, initial_complaint: str, known_info_text: str, 
                                   previous_questions_text: str, medical_guide_content: str) -> str:
        """Get the formatted advanced follow-up question prompt."""
        return cls.ADVANCED_FOLLOWUP_PROMPT.format(
            initial_complaint=initial_complaint,
            known_info_text=known_info_text,
            previous_questions_text=previous_questions_text if previous_questions_text else "- No previous follow-up questions",
            medical_guide_content=medical_guide_content
        )

    @classmethod
    def get_advanced_options_prompt(cls, question: str, medical_guide_content: str) -> str:
        """Get the formatted advanced options prompt."""
        return cls.ADVANCED_OPTIONS_PROMPT.format(
            question=question,
            medical_guide_content=medical_guide_content
        )

    @classmethod
    def get_advanced_examination_prompt(cls, initial_complaint: str, symptoms_summary: str, 
                                      known_info_text: str, previous_exams_text: str, 
                                      medical_guide_content: str) -> str:
        """Get the formatted advanced examination prompt."""
        return cls.ADVANCED_EXAMINATION_PROMPT.format(
            initial_complaint=initial_complaint,
            symptoms_summary=symptoms_summary,
            known_info_text=known_info_text,
            previous_exams_text=previous_exams_text if previous_exams_text else "- No previous examinations",
            medical_guide_content=medical_guide_content
        )

    @classmethod
    def get_referral_summary(cls, initial_complaint: str, key_findings: str) -> str:
        """Get the formatted referral summary."""
        return cls.REFERRAL_SUMMARY_TEMPLATE.format(
            initial_complaint=initial_complaint,
            key_findings=key_findings
        )

    @classmethod
    def get_complete_assessment(cls, diagnosis_text: str, treatment_text: str, 
                              citations_text: str, has_formal_exam: bool = True) -> str:
        """Get the formatted complete assessment."""
        limitation_note = "" if has_formal_exam else cls.LIMITATION_NOTES['no_formal_exam']
        
        return cls.COMPLETE_ASSESSMENT_TEMPLATE.format(
            diagnosis_text=diagnosis_text,
            treatment_text=treatment_text,
            limitation_note=limitation_note,
            citations_text=citations_text
        )

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