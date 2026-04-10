"""
execution_loop.py

This script takes preprocessed patient flags and evaluates them against 
the parsed DMN rules. It filters the rules for a specific 
clinical module, reads them sequentially from top to bottom, and returns 
the outputs of the first rule where all conditions perfectly match the patient. 

Input (patient_flags):
- Patient data, a single unified dictionary where every key maps to:
    1. Booleans: True/False
    2. Unknown?: None (Missing data; triggers the emergency lookahead logic)
Note: This program assumes that all raw patient data (strings, numbers, categories, cont. vars.) 
into these Boolean flags beforehand. Currently, strings (ex. "< 14) are still being evaluated, which is fragile.

Outputs:
- Success: Dictionary of clinical actions and triage. (e.g., {"triage": "hospital", "danger_sign": True})
    - From "outputs" dict of parsed_rules.json
- Halt: Error dictionary identifying which missing data caused the stop.

Logic:
If the patient has the data but it doesn't match the rule's requirement, it just moves on to the next rule.

If a rule requires a specific piece of data and the patient data is missing (Unknown/None),
the engine will halt. It won't just skip the rule and proceed. Prevents dangerous fall-through misdiagnoses.

If data is missing, the engine will temporarily log a "caution flag" and keep scanning the remaining rules. 
If it finds a perfect match further down that triggers an emergency (like a hospital referral), 
it will override the missing data and trigger the emergency immediately. 
If no emergency is found, it will halt and return an error using the caution flag.
"""

import json

def evaluate_decision_module(patient_flags, all_rules, target_module):
    """
    Loops through the rules for a specific module and returns the first perfect match.
    """
    # Filters the rulebook to only include rules for the module we are currently running
    module_rules = []
    for rule in all_rules:
        # Checks if the rule's metadata matches the module we asked for (e.g., "decide_module_a")
        if rule["metadata"]["decision"] == target_module:
            # If it matches, add it to our filtered list
            module_rules.append(rule)

    # Tracks if we hit a missing data roadblock
    missing_data_error = None 

    # Reads the rules row by row (top to bottom)
    for rule in module_rules:
        is_match = True
        is_unknown = False
        missing_condition = None
        
        # Checks every condition required by this specific row
        for condition_name, required_value in rule["conditions"].items():
            
            # Use .get(), if the patient doesn't have the data, it returns None (Unknown)
            # Ensures we never assume missing data equals True or False.
            patient_value = patient_flags.get(condition_name)

            # If the data is completely missing, we can't safely proceed
            if patient_value is None:
                is_unknown = True
                is_match = False
                missing_condition = condition_name
                break # Stop checking conditions for this specific rule
            
            # If patient's value doesn't perfectly match the requirement for the rule, this rule fails
            if patient_value != required_value:
                is_match = False
                break # Stop checking this rule and immediately move to the next row
                
        # Rule Evaluation
        
        # Scenario A: we found a perfect match
        if is_match:
            # Check if this rule is an emergency (hospital triage or danger sign)
            is_emergency = (rule["outputs"].get("triage") == "hospital" or rule["outputs"].get("danger_sign") is True)
            
            # If we previously hit missing data, we can ONLY return this match if it's an emergency
            if missing_data_error and not is_emergency:
                return missing_data_error
            
            # Otherwise (it's an emergency, OR we never had missing data), return the match
            return rule["outputs"]
            
        # Scenario B: we couldn't evaluate this rule due to missing data
        if is_unknown and not missing_data_error:
            # We log the caution flag, but we DON'T return it yet. Keep looping for emergencies!
            missing_data_error = {
                "error": f"Error: Missing data for '{missing_condition}'. Cannot safely evaluate rule."
            }

    # If we loop through the whole table and never found a match or an emergency:
    if missing_data_error:
        return missing_data_error
        
    # Fallback if we check every single rule and none match
    return {"error": f"No matching rule found in {target_module}"}

# Test Setup
if __name__ == "__main__":
    # Load sample rulebook
    try:
        with open('parsed_rules.json', 'r') as f:
            parsed_table = json.load(f)
    except FileNotFoundError:
        print("Error: Make sure 'parsed_rules.json' is in the same folder as this script.")
        exit(1)

    # Dummy Patient, we know they have chest indrawing, but we don't know anything else
    # -> Emergency
    dummy_patient = {
        "chest_indrawing_present": True
    }

    print("Evaluating patient against Module A...")
    
    # Run the evaluator
    result = evaluate_decision_module(dummy_patient, parsed_table, "decide_module_a")
    
    print("\n------------- Clinical Outputs ------------")
    print(json.dumps(result, indent=2))

    # Load dummy patients
    try:
        with open('dummy_patients.json', 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: Make sure 'dummy_patients.json' is in the same folder.")
        exit(1)

    # Loop through the JSON file and test each patient
    for case in test_cases:
        print(f"\n--- {case['test_name']} ---")
        
        result = evaluate_decision_module(
            patient_flags=case['patient_flags'], 
            all_rules=parsed_table, 
            target_module=case['target_module']
        )
        
        print(json.dumps(result, indent=2))