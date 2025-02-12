import streamlit as st
import sys
import os
from dotenv import load_dotenv
import json

# Import only the functions from chad.py, not the global variables
from chad import (
    process_with_matrix, judge, judge_exam, get_diagnosis_and_treatment,
    parse_examination_text, store_examination, parse_question_data,
    extract_question_and_options, extract_examination_and_options,
    get_embedding_batch, vectorQuotesWithSource
)

# Initialize environment variables
load_dotenv()

# Define initial questions (moved from chad.py)
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

def initialize_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'structured_questions_array' not in st.session_state:
        st.session_state.structured_questions_array = []
    if 'examination_history' not in st.session_state:
        st.session_state.examination_history = []
    if 'initial_responses' not in st.session_state:
        st.session_state.initial_responses = []
    if 'followup_responses' not in st.session_state:
        st.session_state.followup_responses = []
    if 'exam_responses' not in st.session_state:
        st.session_state.exam_responses = []

def display_initial_questions():
    """Handle initial questions using Streamlit components"""
    st.subheader("Initial Screening Questions")
    
    all_answered = True
    for i, question in enumerate(questions_init):
        st.write(f"\n{question['question']}")
        
        key_prefix = f"initial_q{i}"
        
        if question['type'] in ['MC', 'YN', 'MCM']:
            if question['type'] == 'MCM':
                selected = st.multiselect(
                    "Select all that apply",
                    [opt['text'] for opt in question['options']],
                    key=f"{key_prefix}_mcm"
                )
                if selected:
                    st.session_state.initial_responses.append({
                        "question": question['question'],
                        "answer": selected,
                        "type": question['type']
                    })
                else:
                    all_answered = False
            else:
                selected = st.selectbox(
                    "Choose one",
                    [opt['text'] for opt in question['options']],
                    key=f"{key_prefix}_mc"
                )
                if selected == "Other (please specify)":
                    custom = st.text_input("Please specify:", key=f"{key_prefix}_custom")
                    if custom:
                        st.session_state.initial_responses.append({
                            "question": question['question'],
                            "answer": custom,
                            "type": question['type']
                        })
                    else:
                        all_answered = False
                elif selected:
                    st.session_state.initial_responses.append({
                        "question": question['question'],
                        "answer": selected,
                        "type": question['type']
                    })
                else:
                    all_answered = False
        
        elif question['type'] == 'NUM':
            value = st.number_input(
                f"Enter a number between {question['range']['min']} and {question['range']['max']}",
                min_value=question['range']['min'],
                max_value=question['range']['max'],
                step=question['range']['step'],
                key=f"{key_prefix}_num"
            )
            st.session_state.initial_responses.append({
                "question": question['question'],
                "answer": value,
                "type": question['type']
            })
        
        elif question['type'] == 'FREE':
            text = st.text_area("Your response:", key=f"{key_prefix}_free")
            if text:
                st.session_state.initial_responses.append({
                    "question": question['question'],
                    "answer": text,
                    "type": question['type']
                })
            else:
                all_answered = False

    if st.button("Continue to Follow-up Questions", disabled=not all_answered):
        st.session_state.step = 1
        st.rerun()

def display_followup_questions():
    """Handle follow-up questions using chad.py's functions"""
    st.subheader("Follow-up Questions")
    
    current_question = extract_question_and_options(
        st.session_state.structured_questions_array, 
        len(st.session_state.structured_questions_array)
    )
    
    if current_question:
        st.write(current_question[0])  # Display question text
        selected = st.radio("Choose one:", current_question[1:])  # Options start from index 1
        
        if selected == "Other (please specify)":
            custom = st.text_input("Please specify:")
            answer = custom
        else:
            answer = selected
        
        if st.button("Submit Answer"):
            st.session_state.followup_responses.append({
                "question": current_question[0],
                "answer": answer,
                "type": "MC"
            })
            
            if judge(st.session_state.followup_responses, current_question[0]):
                st.session_state.step = 2
                st.rerun()

def display_examinations():
    """Handle examinations using chad.py's functions"""
    st.subheader("Medical Examinations")
    
    current_exam = extract_examination_and_options(
        st.session_state.examination_history, 
        len(st.session_state.examination_history)
    )
    
    if current_exam:
        st.write("Recommended Examination:")
        st.write(current_exam[0])
        
        selected = st.radio("Select finding:", current_exam[1:])
        
        if selected == "Other (please specify)":
            custom = st.text_input("Please specify finding:")
            finding = custom
        else:
            finding = selected
        
        if st.button("Submit Finding"):
            st.session_state.exam_responses.append({
                "examination": current_exam[0],
                "result": finding,
                "type": "EXAM"
            })
            
            if judge_exam(st.session_state.exam_responses, current_exam[0]):
                st.session_state.step = 3
                st.rerun()

def display_results():
    """Display final results using chad.py's get_diagnosis_and_treatment"""
    st.title("Medical Assessment Results")
    
    results = get_diagnosis_and_treatment(
        st.session_state.initial_responses,
        st.session_state.followup_responses,
        st.session_state.exam_responses
    )
    
    if results:
        st.header("Diagnosis")
        st.write(results["diagnosis"])
        
        st.header("Recommended Treatment Plan")
        st.write(results["treatment"])
        
        st.header("Key References")
        for citation in results["citations"]:
            st.write(f"- {citation['source']} (relevance: {citation['score']:.2f})")
        
        if st.checkbox("Show Assessment Details"):
            st.json(st.session_state.structured_questions_array)
            st.json(st.session_state.examination_history)
    
    if st.button("Start New Assessment"):
        # Clear all session state
        st.session_state.clear()
        initialize_session_state()
        st.rerun()

def main():
    st.set_page_config(page_title="Medical Assessment System", page_icon="🏥", layout="wide")
    st.title("Medical Assessment System")
    
    initialize_session_state()
    
    progress_labels = ["Initial Questions", "Follow-up Questions", "Examinations", "Results"]
    progress = st.session_state.step / (len(progress_labels) - 1)
    st.progress(progress)
    st.write(f"Current Step: {progress_labels[st.session_state.step]}")
    
    if st.session_state.step == 0:
        display_initial_questions()
    elif st.session_state.step == 1:
        display_followup_questions()
    elif st.session_state.step == 2:
        display_examinations()
    elif st.session_state.step == 3:
        display_results()
    
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("Global Arrays Status:")
        st.sidebar.json({
            "structured_questions": len(st.session_state.structured_questions_array),
            "examination_history": len(st.session_state.examination_history),
            "initial_responses": len(st.session_state.initial_responses),
            "followup_responses": len(st.session_state.followup_responses),
            "exam_responses": len(st.session_state.exam_responses)
        })

if __name__ == "__main__":
    main()