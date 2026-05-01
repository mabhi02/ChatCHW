"""Run product-team JSON logic for DMN-side black-box testing.
Loads patient inputs, executes modules, and writes structured execution logs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGIC_PATH = SCRIPT_DIR / "clinical_logic.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "product_dmn_sample_logs.json"
DEFAULT_DMN_PATIENTS_PATH = SCRIPT_DIR / "dmn_ready_sample_patients.json"
DEFAULT_ANGELINA_OUTPUT = SCRIPT_DIR.parent / "Angelina" / "dmn" / "output.json"
DEFAULT_ANGELINA_PREPROCESSOR = SCRIPT_DIR.parent / "Angelina" / "dmn" / "dmn_preprocessor.py"


def load_artifacts(logic_path: Path) -> Dict[str, Any]:
    with logic_path.open(encoding="utf-8") as f:
        logic = json.load(f)
    required_keys = {"modules", "router"}
    missing = sorted(required_keys - set(logic.keys()))
    if missing:
        raise ValueError(f"Logic JSON missing required keys: {missing}")
    if not isinstance(logic["router"], dict) or "rows" not in logic["router"]:
        raise ValueError("Logic JSON must contain router.rows")
    return logic


def translate_expr(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace('&&', ' and ').replace('||', ' or ')
    expr = re.sub(r'\bAND\b', ' and ', expr)
    expr = re.sub(r'\bOR\b', ' or ', expr)
    # replace ! but not !=
    expr = re.sub(r'(?<![=!])!(?!=)', ' not ', expr)
    expr = re.sub(r'\btrue\b', 'True', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bfalse\b', 'False', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bnull\b', 'None', expr, flags=re.IGNORECASE)
    return expr


def eval_expr(expr: str, state: Dict[str, Any]) -> bool:
    py_expr = translate_expr(expr)
    safe_globals = {'__builtins__': {}}
    try:
        return bool(eval(py_expr, safe_globals, defaultdict_false(state)))
    except Exception:
        # Fail closed for MVP
        return False


class defaultdict_false(dict):
    def __missing__(self, key):
        return False


def execute_module(module: Dict[str, Any], state: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    for idx, rule in enumerate(module.get('rules', [])):
        if eval_expr(rule['condition'], state):
            outputs = deepcopy(rule.get('outputs', {}))
            # Some outputs are references to variable names; resolve them if present in state.
            for k, v in list(outputs.items()):
                if isinstance(v, str) and v in state:
                    outputs[k] = state[v]
            state.update(outputs)
            state[module.get('done_flag', f"{module['module_id']}_done")] = True
            return {
                'rule_id': rule.get('rule_id'),
                'rule_index': idx,
                'outputs': outputs,
                'description': rule.get('description'),
            }, state
    # no match: mark done to avoid infinite loop
    state[module.get('done_flag', f"{module['module_id']}_done")] = True
    return None, state


def infer_module_result(executed: Optional[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    outputs = executed['outputs'] if executed else {}
    triage = 'home'
    # infer triage from referral signals
    ref_dest = outputs.get('ref_destination') or outputs.get('ref_referral_destination') or outputs.get('priority_exit_destination')
    if ref_dest in ('hospital', 'urgent_referral_hospital'):
        triage = 'hospital'
    elif ref_dest in ('health_facility', 'clinic') or outputs.get('ref_urgently') or outputs.get('ref_urgent_referral'):
        triage = 'clinic'
    danger = bool(
        outputs.get('p_danger_sign_present')
        or outputs.get('is_priority_exit')
        or any(k.startswith('p_danger_sign_') and bool(v) for k, v in outputs.items())
    )
    reason = (
        outputs.get('ref_reason')
        or outputs.get('ref_referral_reason')
        or outputs.get('description')
        or (executed.get('description') if executed else None)
        or ''
    )
    return {
        'triage': triage,
        'danger_sign': danger,
        'reason': reason,
        'matched_rule_index': executed['rule_index'] if executed else None,
    }


def final_outcome_from_state(state: Dict[str, Any], module_results: Dict[str, Any]) -> Dict[str, Any]:
    # prefer last non-home module result
    triage = 'home'
    reason = ''
    for m in module_results.values():
        if m.get('triage') == 'hospital':
            triage = 'hospital'; reason = m.get('reason',''); break
        if m.get('triage') == 'clinic' and triage != 'hospital':
            triage = 'clinic'; reason = m.get('reason','')
    danger = bool(state.get('p_danger_sign_present') or state.get('is_priority_exit'))
    clinic_referral = triage == 'clinic'
    ref = state.get('ref_destination') or state.get('ref_referral_destination') or state.get('priority_exit_destination') or ''
    actions = []
    for k, v in state.items():
        if isinstance(v, bool) and v and (k.startswith('rx_') or k.startswith('adv_') or k.startswith('fu_')):
            actions.append(k)
        elif isinstance(v, (str, int, float)) and k.startswith('rx_') and v not in ('', None, False):
            actions.append(f'{k}={v}')
    return {
        'triage': triage,
        'danger_sign': danger,
        'clinic_referral': clinic_referral,
        'reason': reason,
        'ref': ref,
        'actions': '; '.join(actions[:12]),
    }


def run_patient(patient_id: str, patient_inputs: Dict[str, Any], logic: Dict[str, Any]) -> Dict[str, Any]:
    modules = logic['modules']
    router = logic['router']['rows']
    state = defaultdict_false()
    state.update(deepcopy(patient_inputs))
    state['sys_encounter_start'] = True
    # initialize done flags false
    for m in modules.values():
        state[m['done_flag']] = False
    module_results: Dict[str, Any] = {}
    executed_order: List[str] = []
    for _ in range(30):
        next_module = None
        for row in sorted(router, key=lambda r: r['priority']):
            if eval_expr(row['condition'], state):
                next_module = row['output_module']
                break
        if not next_module:
            break
        if next_module == 'PRIORITY_EXIT':
            break
        if next_module == 'mod_home_care_advice' and state.get('mod_home_care_advice_done'):
            break
        mod = modules.get(next_module)
        if not mod:
            break
        executed, state = execute_module(mod, state)
        module_results[next_module] = infer_module_result(executed, state)
        executed_order.append(next_module)
        if next_module == 'mod_home_care_advice':
            break
    log = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'patient_id': patient_id,
        'inputs': patient_inputs,
        'module_results': module_results,
        'final_outcome': final_outcome_from_state(state, module_results),
        'execution_trace': executed_order,
    }
    return log


def built_in_sample_patients() -> List[Tuple[str, Dict[str, Any]]]:
    base = {
        'demo_age_months': 24,
        'demo_age_in_weeks': 104,
        'demo_sex': 'male',
        'demo_child_name': 'Sample Child',
        'demo_caregiver_name': 'Caregiver',
        'demo_weight_kg': 10.5,
        'demo_village': 'Test Village',
        'q_cough_present': False,
        'q_cough_duration_days': 0,
        'q_cough_days': 0,
        'q_has_diarrhoea': False,
        'q_diarrhoea_present': False,
        'q_diarrhoea_duration_days': 0,
        'q_diarrhoea_days': 0,
        'q_blood_in_stool': False,
        'q_has_fever': False,
        'q_fever_present': False,
        'q_fever_duration_days': 0,
        'q_fever_days': 0,
        'q_convulsions_present': False,
        'hx_convulsions': False,
        'q_able_to_drink_feed': True,
        'ex_not_able_to_drink_or_feed': False,
        'q_vomits_everything': False,
        'ex_vomits_everything': False,
        'ex_chest_indrawing': False,
        'ex_unusually_sleepy_unconscious': False,
        'ex_unusually_sleepy_or_unconscious': False,
        'ex_bilateral_foot_oedema': False,
        'ex_swelling_both_feet': False,
        'ex_respiratory_rate_bpm': 30,
        'v_muac_mm': 135,
        'v_muac_category': 'green',
        'hx_vaccine_card_present': False,
        'hx_bcg_given': False,
        'hx_hepb_birth_given': False,
        'hx_opv0_given': False,
        'p_malaria_endemic_area': False,
        'p_malaria_area': False,
        'hx_malaria_area': False,
        'lab_rdt_result': 'not_done',
        'p_rdt_positive': False,
        'p_any_danger_sign': False,
        'p_danger_sign_present': False,
        'p_fast_breathing': False,
    }
    p1 = deepcopy(base)
    p1.update({
        'q_cough_present': True,
        'q_cough_duration_days': 3,
        'q_cough_days': 3,
        'ex_respiratory_rate_bpm': 52,
    })
    p2 = deepcopy(base)
    p2.update({
        'q_has_diarrhoea': True,
        'q_diarrhoea_present': True,
        'q_diarrhoea_duration_days': 2,
        'q_diarrhoea_days': 2,
        'q_has_fever': True,
        'q_fever_present': True,
        'q_fever_duration_days': 2,
        'q_fever_days': 2,
        'lab_rdt_result': 'positive',
        'p_rdt_positive': True,
        'hx_malaria_area': True,
        'p_malaria_area': True,
        'p_malaria_endemic_area': True,
    })
    return [('sample_cough_fast_breathing', p1), ('sample_diarrhoea_fever', p2)]


def _normalize_patient_record(record: Dict[str, Any], idx: int) -> Tuple[str, Dict[str, Any]]:
    if "patient_id" in record and "inputs" in record and isinstance(record["inputs"], dict):
        return str(record["patient_id"]), dict(record["inputs"])
    patient_id = (
        record.get("patient_id")
        or (f"row_{record['_row_number']}" if "_row_number" in record else f"patient_{idx}")
    )
    return str(patient_id), dict(record)


def load_dmn_ready_patients(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("DMN-ready patient input must be a JSON list")
    return [_normalize_patient_record(record, idx) for idx, record in enumerate(rows, start=1)]


def load_patients_via_angelina_preprocessor(
    output_json_path: Path,
    preprocessor_path: Path,
) -> List[Tuple[str, Dict[str, Any]]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("angelina_dmn_preprocessor", preprocessor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preprocessor module from {preprocessor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "batch_preprocess"):
        raise AttributeError("Angelina preprocessor is missing batch_preprocess(path, context=None)")
    rows: Iterable[Dict[str, Any]] = module.batch_preprocess(str(output_json_path))
    normalized = [_normalize_patient_record(record, idx) for idx, record in enumerate(rows, start=1)]
    return [(pid, adapt_angelina_inputs_for_product_logic(inputs)) for pid, inputs in normalized]


def _muac_category_to_mm(category: Any) -> int:
    cat = str(category or "").strip().lower()
    if cat == "red":
        return 110
    if cat == "yellow":
        return 120
    return 130


def adapt_angelina_inputs_for_product_logic(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Map Angelina preprocessor keys to this product clinical_logic key shape."""
    age_months = int(inputs.get("demo_child_age_months", 0))
    rr_estimate = int(inputs.get("v_respiratory_rate_per_min", 0))
    ex_not_able = bool(inputs.get("q_not_able_to_drink_or_feed", False))
    q_able = not ex_not_able
    q_vomit = bool(inputs.get("q_vomits_everything", False))
    cough_days = int(inputs.get("q_cough_duration_days", 0) or 0)
    diarrhoea_days = int(inputs.get("q_diarrhoea_duration_days", 0) or 0)
    fever_days = int(inputs.get("q_fever_duration_days", 0) or 0)
    malaria = bool(inputs.get("hx_malaria_area", False))
    rdt_result = str(inputs.get("lab_rdt_malaria_result", "not_done"))
    muac_category = str(inputs.get("v_muac_strap_colour", "green"))
    swelling = bool(inputs.get("ex_swelling_both_feet", False))
    sleepy = bool(inputs.get("ex_unusually_sleepy_or_unconscious", False))
    convulsions = bool(inputs.get("q_convulsions", False))
    mapped = {
        "demo_age_months": age_months,
        "q_cough_present": bool(inputs.get("q_has_cough", False)),
        "q_cough_duration_days": cough_days,
        "q_cough_days": cough_days,
        "q_has_diarrhoea": bool(inputs.get("q_has_diarrhoea", False)),
        "q_diarrhoea_present": bool(inputs.get("q_has_diarrhoea", False)),
        "q_diarrhoea_duration_days": diarrhoea_days,
        "q_diarrhoea_days": diarrhoea_days,
        "q_blood_in_stool": bool(inputs.get("q_blood_in_stool", False)),
        "q_has_fever": bool(inputs.get("q_has_fever", False)),
        "q_fever_present": bool(inputs.get("q_has_fever", False)),
        "q_fever_duration_days": fever_days,
        "q_fever_days": fever_days,
        "q_convulsions_present": convulsions,
        "hx_convulsions": convulsions,
        "q_able_to_drink_feed": q_able,
        "ex_not_able_to_drink_or_feed": ex_not_able,
        "q_vomits_everything": q_vomit,
        "ex_vomits_everything": q_vomit,
        "ex_chest_indrawing": bool(inputs.get("ex_chest_indrawing", False)),
        "ex_unusually_sleepy_unconscious": sleepy,
        "ex_unusually_sleepy_or_unconscious": sleepy,
        "ex_bilateral_foot_oedema": swelling,
        "ex_swelling_both_feet": swelling,
        "ex_respiratory_rate_bpm": rr_estimate,
        "v_muac_category": muac_category,
        "v_muac_mm": _muac_category_to_mm(muac_category),
        "p_malaria_endemic_area": malaria,
        "p_malaria_area": malaria,
        "hx_malaria_area": malaria,
        "lab_rdt_result": rdt_result,
        "lab_rdt_malaria_result": rdt_result,
        "p_rdt_positive": (rdt_result == "positive"),
        "_row_number": inputs.get("_row_number"),
    }
    return mapped


def write_logs(logs: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run product-team DMN JSON logic and emit structured logs")
    parser.add_argument("--logic", type=Path, default=DEFAULT_LOGIC_PATH, help="Path to clinical_logic.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output log JSON")
    parser.add_argument(
        "--patients-source",
        choices=("dmn_ready_json", "angelina_output", "built_in"),
        default="dmn_ready_json",
        help="How to load patient inputs",
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=DEFAULT_DMN_PATIENTS_PATH,
        help="Path to DMN-ready patient JSON list",
    )
    parser.add_argument(
        "--angelina-output-path",
        type=Path,
        default=DEFAULT_ANGELINA_OUTPUT,
        help="Path to Angelina output.json",
    )
    parser.add_argument(
        "--angelina-preprocessor-path",
        type=Path,
        default=DEFAULT_ANGELINA_PREPROCESSOR,
        help="Path to Angelina dmn_preprocessor.py",
    )
    parser.add_argument("--max-patients", type=int, default=0, help="Optional cap for number of patients")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.logic.exists():
        print(
            f"Missing logic artifact: {args.logic}\n"
            "Place the product-team clinical_logic.json here or pass --logic <path>.",
            file=sys.stderr,
        )
        return 2

    logic = load_artifacts(args.logic)
    if args.patients_source == "dmn_ready_json":
        patients = load_dmn_ready_patients(args.patients_path)
    elif args.patients_source == "angelina_output":
        patients = load_patients_via_angelina_preprocessor(
            args.angelina_output_path,
            args.angelina_preprocessor_path,
        )
    else:
        patients = built_in_sample_patients()

    if args.max_patients and args.max_patients > 0:
        patients = patients[: args.max_patients]
    logs = [run_patient(pid, patient, logic) for pid, patient in patients]
    write_logs(logs, args.output)
    print(f"Wrote {len(logs)} logs to {args.output}")
    if logs:
        print(json.dumps(logs[0], indent=2)[:3000])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
