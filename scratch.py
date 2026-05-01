import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Testing.Gigi.medical_team.medical_team_runner import adapt_inputs, matches, load_modules

with open("Testing/Gigi/product_dmn_sample_logs_fixed.json") as f:
    logs = json.load(f)
    row_2 = next(row for row in logs if row["patient_id"] == "row_2")

p_in = row_2["inputs"]

MODULE_FILES = [
    "Testing/Gigi/medical_team/normalized_rules/diarrhea.json",
    "Testing/Gigi/medical_team/normalized_rules/pneumonia_cough.json",
    "Testing/Gigi/medical_team/normalized_rules/fever_malaria.json"
]

modules = load_modules(MODULE_FILES)
adapted = adapt_inputs(p_in)
print("ADAPTED INPUTS:")
print(json.dumps(adapted, indent=2))

for mod in modules:
    print(f"\nChecking module: {mod.get('module_id')}")
    for rule in mod.get("rules", []):
        match = True
        for k, expected in rule.get("conditions", {}).items():
            if k not in adapted:
                expected_upper = str(expected).upper()
                if expected_upper != "ANY" and expected_upper != "N/A" and "OR ANY" not in expected_upper:
                    print(f"  Rule {rule.get('rule_index')} failed on missing key: {k}")
                    match = False
                    break
                continue
            if not matches(expected, adapted[k]):
                print(f"  Rule {rule.get('rule_index')} failed on condition: {k} (expected '{expected}', got '{adapted[k]}')")
                match = False
                break
        if match:
            print(f"==> Matched Rule {rule.get('rule_index')}")
            outs = rule.get("outputs", {})
            print("Outputs:", json.dumps(outs, indent=2))
            for ok, ov in outs.items():
                adapted[ok] = str(ov)
            break
