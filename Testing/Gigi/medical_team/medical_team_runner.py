# Testing/Gigi/medical_team/medical_team_runner.py
# Minimal medical-side rule interpreter to produce comparable outputs for MVP testing.
import json
import os

def adapt_inputs(p_in):
    out = {}
    
    out["q_stools_loose_or_watery"] = "YES" if p_in.get("q_diarrhoea_present") else "NO"
    out["q_blood_in_stool"] = "YES" if p_in.get("q_blood_in_stool") else "NO"
    out["ex_convulsions"] = "YES" if p_in.get("hx_convulsions") else "NO"
    out["q_convulsions_this_illness"] = "YES" if p_in.get("hx_convulsions") else "NO"
    
    vomits = p_in.get("q_vomits_everything") or p_in.get("ex_vomits_everything")
    out["q_vomits_everything?"] = "YES" if vomits else "NO"
    out["q_vomits_everything"] = "YES" if vomits else "NO"
    
    out["ex_chest_indrawing"] = "YES" if p_in.get("ex_chest_indrawing") else "NO"
    
    muac_red = p_in.get("v_muac_category") == "red"
    out["lab_muac_is_red?"] = "YES" if muac_red else "NO"
    out["ex_muac_color_result"] = "RED" if muac_red else "NOT RED"
    
    swelling = p_in.get("ex_swelling_both_feet")
    out["ex_swelling_on_feet"] = "YES" if swelling else "NO"
    out["ex_feet_swelling_bilateral"] = "YES" if swelling else "NO"
    
    sleepy = p_in.get("ex_unusually_sleepy_unconscious") or p_in.get("ex_unusually_sleepy_or_unconscious")
    out["ex_sleepy_unconcious"] = "YES" if sleepy else "NO"
    out["ex_unusually_sleepy_or_unconscious"] = "YES" if sleepy else "NO"
    
    fever = p_in.get("q_fever_present")
    out["ex_fever_present"] = "YES" if fever else "NO"
    out["q_fever_now_or_last_3d"] = "YES" if fever else "NO"
    
    fever_dur = p_in.get("q_fever_duration_days", 0)
    out["q_duration_fever_days"] = ">=7" if fever_dur >= 7 else "<7"
    out["q_fever_days_since_start_days"] = ">=7" if fever_dur >= 7 else "<7"
    
    diar_dur = p_in.get("q_diarrhoea_duration_days", 0)
    out["q_duration_loos_stools_days"] = ">=14" if diar_dur >= 14 else "<14"
    out["q_diarrhea_duration_days"] = ">=14" if diar_dur >= 14 else "<14"
    
    cough_dur = p_in.get("q_cough_duration_days", 0)
    out["q_cough_duration_days"] = ">=14" if cough_dur >= 14 else "<14"
    
    out["q_malaria_area"] = "YES" if p_in.get("p_malaria_endemic_area") else "NO"
    
    can_drink = p_in.get("q_able_to_drink_feed")
    out["ex_able_to_drink_eat"] = "YES" if can_drink else "NO"
    out["q_not_able_drink_or_eat"] = "NO" if can_drink else "YES"
    out["q_can_drink"] = "YES" if can_drink else "NO"
    
    age = p_in.get("demo_age_months", 0)
    rr = p_in.get("ex_respiratory_rate_bpm", 0)
    is_fast_breathing = False
    if age < 12 and rr >= 50: is_fast_breathing = True
    elif 12 <= age < 60 and rr >= 40: is_fast_breathing = True
    out["v_resp_rate_per_min"] = "AGE-THRESHOLD" if is_fast_breathing else "BELOW THRESHOLD"
    
    # basic age bands used in conditions
    if 2 <= age <= 5: out["demo_age_months"] = "2-5 MONTHS"
    elif 6 <= age <= 59: out["demo_age_months"] = "6-59 MONTHS"
    else: out["demo_age_months"] = "OTHER"
    
    return out

def matches(condition_val, actual_val):
    if condition_val is None: return True
    cv = str(condition_val).upper().strip()
    av = str(actual_val).upper().strip()
    
    if cv == "ANY" or cv == "N/A": return True
    
    if "/ANY" in cv:
        cv = cv.split("/ANY")[0].strip()
        
    if cv == "ANY" or cv == "N/A": return True
    
    if "/ANY" in cv:
        cv = cv.split("/ANY")[0].strip()
        
    if "OR ANY" in cv:
        return True
        
    if cv == av: return True
    
    if "2-59" in cv or "2–59" in cv:
        if "2-5" in av or "6-59" in av:
            return True
    
    if "YES" in cv and "NO" not in cv and av == "YES": return True
    if "NO" in cv and "YES" not in cv and av == "NO": return True
    
    if "NOT RED" in cv and av == "NOT RED": return True
    if "RED" in cv and "NOT RED" not in cv and av == "RED": return True
    
    if "<14" in cv and "<14" in av: return True
    if "≥14" in cv and ">=14" in av: return True
    if ">=14" in cv and ">=14" in av: return True
    if ">14" in cv and ">=14" in av: return True
    
    if "<7" in cv and "<7" in av: return True
    if "≥7" in cv and ">=7" in av: return True
    if ">=7" in cv and ">=7" in av: return True
    
    if "AGE-THRESHOLD" in cv and av == "AGE-THRESHOLD": return True
    if "BELOW THRESHOLD" in cv and av == "BELOW THRESHOLD": return True
    
    if av in cv: return True
    
    return False

def load_modules(module_files):
    modules = []
    for f in module_files:
        if os.path.exists(f):
            with open(f, 'r') as fh:
                data = json.load(fh)
                modules.extend(data.get("modules", []))
    return modules

def run_medical_logic(patient_inputs, module_files):
    modules = load_modules(module_files)
    adapted = adapt_inputs(patient_inputs)
    
    danger_sign = False
    referral = False
    matched_rules = []
    actions = []
    reasons = []
    
    for mod in modules:
        for rule in mod.get("rules", []):
            match = True
            for k, expected in rule.get("conditions", {}).items():
                if k not in adapted:
                    expected_upper = str(expected).upper()
                    if expected_upper != "ANY" and expected_upper != "N/A" and "OR ANY" not in expected_upper:
                        match = False
                        break
                    continue
                if not matches(expected, adapted[k]):
                    match = False
                    break
            if match:
                outs = rule.get("outputs", {})
                # Add outputs to adapted so downstream modules can use them
                for ok, ov in outs.items():
                    adapted[ok] = str(ov)
                
                ds_keys = ["p_danger_sign", "p_danger_sign_present"]
                for dsk in ds_keys:
                    val = str(outs.get(dsk, "")).upper()
                    if "YES" in val:
                        danger_sign = True
                        
                ref_keys = ["ref_health_facility", "ref_health_facility_urgent", "ref_health_facility_immediate", "out_disposition"]
                for rfk in ref_keys:
                    val = str(outs.get(rfk, "")).upper()
                    if "YES" in val or "REFER" in val:
                        referral = True
                
                # Capture reason from Citation
                if "Citation" in outs and outs["Citation"]:
                    reasons.append(outs["Citation"])
                
                # Capture action flags (rx_, adv_)
                for ok, ov in outs.items():
                    if (ok.startswith("rx_") or ok.startswith("adv_") or ok.startswith("tx_")) and str(ov).upper() not in ("N/A", "NO", "FALSE", "NONE", ""):
                        actions.append(f"{ok}={ov}")
                
                matched_rules.append(rule.get("rule_index"))
                break # first match per module
                
    if danger_sign:
        triage = "hospital"
    elif referral:
        triage = "clinic"
    else:
        triage = "home"
        
    return {
        "triage": triage,
        "danger_sign": danger_sign,
        "reason": " | ".join(reasons),
        "actions": "; ".join(actions[:12]),
        "matched_rules_count": len(matched_rules)
    }
