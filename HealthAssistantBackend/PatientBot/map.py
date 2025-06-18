import pandas as pd
import json
import openai
import os
from typing import Dict, List
from dotenv import load_dotenv

def create_case_string(row: pd.Series) -> str:
    """Create a string representation of a case from a row in data.csv"""
    case = f"Age: {row['Age']} ({row['age (years)']} years), "
    case += f"Sex: {row['Sex']}, "
    case += f"Complaint: {row['Complaint']}, "
    case += f"Duration: {row['Duration']}, "
    case += f"CHW Questions: {row['CHW Questions']}, "
    case += f"Exam Findings: {row['Exam Findings']}"
    return case

def extract_rubric_cases(rubric_df: pd.DataFrame) -> Dict[int, str]:
    """Extract cases from the rubric file and create a mapping of row numbers to case descriptions"""
    cases = {}
    for idx, row in rubric_df.iterrows():
        cases[idx + 1] = row['True diagnosis ']
    return cases

def get_openai_mapping(case_string: str, rubric_cases: Dict[int, str]) -> int:
    """Use OpenAI API to map a case to a rubric row"""
    prompt = f"Based on this case:\n{case_string}\n\nWhich of the following cases does it correspond to?\n{json.dumps(rubric_cases, indent=2)}\n\nONLY output the number of the row that the case corresponds to."
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that maps medical cases to their corresponding rubric rows. Only output the row number."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        # Extract the row number from the response
        row_number = int(response.choices[0].message.content.strip())
        return row_number
    except Exception as e:
        print(f"Error processing case: {e}")
        return None

def main():
    # Read the CSV files
    data_df = pd.read_csv("data.csv", usecols=['Age', 'age (years)', 'Sex', 'Complaint', 'Duration', 'CHW Questions', 'Exam Findings'])
    rubric_df = pd.read_csv("Grading rubric WHO 2012 ChatGPT (1).xlsx - Rubric  (1).csv", usecols=['True diagnosis '])
    
    # Extract rubric cases
    rubric_cases = extract_rubric_cases(rubric_df)
    
    # Process each row in data.csv
    mappings = []
    for idx, row in data_df.iterrows():
        print(f"Processing case {idx + 1}/{len(data_df)}")
        case_string = create_case_string(row)
        mapping = get_openai_mapping(case_string, rubric_cases)
        mappings.append(mapping)
    
    # Add mappings as a new column
    data_df['Mapped_Rubric_Row'] = mappings
    
    # Save the updated dataframe
    data_df.to_csv("mapped_data.csv", index=False)
    print("Mapping complete. Results saved to mapped_data.csv")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY in your .env file")
    
    openai.api_key = api_key
    main() 