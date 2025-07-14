# Grading Logic for `run_grade_case.py`

## Overview

`run_grade_case.py` is an interactive script designed to:
- Run a single patient case through the full ChatCHW workflow (initial questions, follow-ups, exams, diagnosis)
- Grade the results using a rubric and GPT-4
- Output both a detailed JSON and a human-friendly CSV summary for review

---

## Workflow Steps

### 1. Case Selection
- Displays a list of available patient cases from `data.csv`.
- User selects a case by its ID.

### 2. Case Loading
- Loads the selected patient case from the PatientBot API.

### 3. Full ChatCHW Workflow
- **Initial Questions:**
  - Asks all required initial questions.
  - Auto-selects answers using PatientBot and AI logic.
- **Follow-up Questions:**
  - Dynamically generates and asks follow-up questions based on the initial complaint and context.
  - Continues up to a maximum or until sufficient information is gathered.
- **Examination Phase:**
  - Proceeds through examination steps, asking for findings and storing results.
- **Diagnosis:**
  - Generates a final diagnosis and treatment plan using the collected responses.

### 4. Rubric Row Selection (Automated)
- Reads `mapped_data.csv` to automatically select the correct rubric row for the case, based on the `Mapped_Rubric_Row` column.
- If the mapping is missing or invalid, prompts the user for manual selection.

### 5. Grading
- For each rubric aspect (e.g., diagnosis, treatment, required questions), uses GPT-4 to compare the case results to the rubric and assigns a score (`1`, `0.5`, or `0`).
- All grading results are collected for output.

### 6. Output Generation
- **JSON Output:**
  - Saves all case results and grading details as a JSON file.
- **CSV Output:**
  - Uses GPT-4 to format the case as a human-readable CSV table with columns:
    - Section
    - ChatCHW asks
    - Reply
    - GraderBot rubric
    - Similarity
  - The format matches the provided example screenshot.

### 7. Organized Output Folder
- Each run creates a new subfolder under `7/14-outputs/` named with the case number and timestamp:

  ```
  7/14-outputs/
    └── Case{case_id}-{timestamp}/
        ├── Case{case_id}-{timestamp}-output.json
        └── Case{case_id}-{timestamp}-graded.csv
  ```
- Both the JSON and CSV files are saved in this folder for easy review and organization.

---

## Key Features

- **Fully Automated Workflow:** Runs through all steps (not just initial assessment).
- **Automated Rubric Mapping:** Uses `mapped_data.csv` to select the correct rubric row.
- **AI-Assisted Grading:** Uses GPT-4 for both grading and CSV formatting.
- **Dual Output:** Produces both a detailed JSON and a human-friendly CSV for each case.
- **Organized Storage:** Outputs are neatly organized by case and timestamp.

---

## Example Output Structure

```
7/14-outputs/
  └── Case0-07-14-12-34-56/
      ├── Case0-07-14-12-34-56-output.json
      └── Case0-07-14-12-34-56-graded.csv
```

---

## In Short

`run_grade_case.py` is a one-stop tool for running, grading, and reviewing a single patient case, producing both machine- and human-readable outputs, and automating as much of the process as possible. 