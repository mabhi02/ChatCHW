# Testing/Gigi/compare_product_vs_medical.py
# Comparison script to run MVP medical logic and compare triage against product DMN outcomes.
import json
import os
import sys

# Update path to import runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from medical_team.medical_team_runner import run_medical_logic

PRODUCT_LOGS = os.path.join(os.path.dirname(__file__), "product_dmn_sample_logs.json")
MODULE_FILES = [
    os.path.join(os.path.dirname(__file__), "medical_team", "normalized_rules", "diarrhea.json"),
    os.path.join(os.path.dirname(__file__), "medical_team", "normalized_rules", "pneumonia_cough.json"),
    os.path.join(os.path.dirname(__file__), "medical_team", "normalized_rules", "fever_malaria.json")
]
MEDICAL_OUT_FILE = os.path.join(os.path.dirname(__file__), "medical_team", "medical_team_sample_logs.json")
SUMMARY_OUT_FILE = os.path.join(os.path.dirname(__file__), "comparison_summary.txt")

def main():
    if not os.path.exists(PRODUCT_LOGS):
        print(f"Product logs not found at {PRODUCT_LOGS}")
        return
        
    with open(PRODUCT_LOGS, 'r') as f:
        product_data = json.load(f)
        
    medical_logs = []
    matches = 0
    mismatches = 0
    mismatch_examples = []
    
    detailed_comparisons = []
    
    for row in product_data:
        p_in = row["inputs"]
        p_id = row.get("patient_id", "unknown")
        prod_outcome = row.get("final_outcome", {})
        
        prod_triage = prod_outcome.get("triage", "unknown")
        prod_danger_sign = prod_outcome.get("danger_sign", False)
        prod_reason = prod_outcome.get("reason", "")
        prod_actions = prod_outcome.get("actions", "")
        
        med_out = run_medical_logic(p_in, MODULE_FILES)
        med_triage = med_out["triage"]
        med_danger_sign = med_out["danger_sign"]
        
        overall_match = (prod_triage == med_triage)
        
        detailed_comparisons.append({
            "patient_id": p_id,
            "inputs": p_in,
            "product_outcome": prod_outcome,
            "medical_outcome": med_out,
            "field_comparisons": {
                "triage_match": prod_triage == med_triage,
                "danger_sign_match": prod_danger_sign == med_danger_sign
            },
            "overall_match": overall_match
        })
        
        if overall_match:
            matches += 1
        else:
            mismatches += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append({
                    "patient_id": p_id,
                    "product_triage": prod_triage,
                    "medical_triage": med_triage,
                    "matched_rules_count": med_out.get("matched_rules_count", 0)
                })
                
    COMPARISON_OUT_FILE = os.path.join(os.path.dirname(__file__), "detailed_comparison_results.json")
    with open(COMPARISON_OUT_FILE, 'w') as f:
        json.dump(detailed_comparisons, f, indent=2)
        
    with open(SUMMARY_OUT_FILE, 'w') as f:
        f.write("=== PRODUCT VS MEDICAL TRIAGE COMPARISON ===\n")
        f.write(f"Total Patients: {len(product_data)}\n")
        f.write(f"Matches: {matches}\n")
        f.write(f"Mismatches: {mismatches}\n\n")
        f.write("Mismatch Examples (up to 5):\n")
        for ex in mismatch_examples:
            f.write(f"- Patient: {ex['patient_id']} | Product: {ex['product_triage']} | Medical: {ex['medical_triage']} | Rules Matched: {ex['matched_rules_count']}\n")
            
    print(f"Comparison complete. Summary written to {SUMMARY_OUT_FILE}")

if __name__ == "__main__":
    main()
