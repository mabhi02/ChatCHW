import json
import requests

CHT_URL = "https://127-0-0-1.local-ip.medicmobile.org:10443"
AUTH = ("medic", "password")

# Set to True once the /api/v2/records endpoint is working
LIVE_MODE = False

def submit_to_cht(payload):
    """Posts patient record to CHT API. Returns (response_dict, status_code)."""
    try:
        response = requests.post(
            f"{CHT_URL}/api/v1/records",
            json=payload,
            auth=AUTH,
            verify=False,
            timeout=15
        )
        if response.status_code in [200, 201]:
            return response.json(), response.status_code
        else:
            return {"success": False, "error": response.text}, response.status_code
    except Exception as e:
        return {"success": False, "error": str(e)}, 0


def get_report(report_id):
    """Fetches the processed report using the ID returned from submission."""
    try:
        response = requests.get(
            f"{CHT_URL}/medic/{report_id}",
            auth=AUTH,
            verify=False,
            timeout=15
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def submit_dummy(patient):
    """Returns the dummy API response provided for testing."""
    return {"success": True, "id": "abc123def456"}, 200


def get_report_dummy(report_id):
    """Returns a dummy report with triage fields for testing."""
    return {
        "fields": {
            "final_triage": "CLINIC",
            "triage_reason_code": "FAST_BREATHING",
            "final_advice": "Refer to Clinic for suspected severe pneumonia."
        }
    }

def main():
    # Load patient data
    try:
        with open('output.json', 'r') as f:
            patients = json.load(f)
    except FileNotFoundError:
        print("Error: 'output.json' not found.")
        return

    # Pick live/dummy functions
    do_submit = submit_to_cht if LIVE_MODE else submit_dummy
    do_fetch = get_report if LIVE_MODE else get_report_dummy

    execution_log = []
    print(f"Starting CHT sync for {len(patients)} patients... (LIVE_MODE={LIVE_MODE})")

    for patient in patients:
        row_id = patient.get("_row_number", "Unknown")

        # Build the payload CHT expects
        payload = {
            "_meta": {
                "form": "orchestrator",
                "locale": "en"
            },
            **patient
        }

        # Submit to CHT/dummy
        submit_res, status_code = do_submit(payload)

        if submit_res.get("success") and "id" in submit_res:
            report_id = submit_res["id"]

            # Fetch the triage result
            report_data = do_fetch(report_id)
            fields = report_data.get("fields", {})

            # Check if CHT returned triage fields — if any are missing, log alert
            missing_field = None
            for required in ["final_triage", "triage_reason_code", "final_advice"]:
                if required not in fields or fields[required] is None:
                    missing_field = required
                    break

            if missing_field:
                alert_entry = {
                    "patient_row": row_id,
                    "alert": f"Missing data — patient record has no matching field for {missing_field}. Patient skipped."
                }
                execution_log.append(alert_entry)
                print(f"Row {row_id}: Missing data alert — no {missing_field}.")
                continue

            # Log successful triage
            log_entry = {
                "patient_row": row_id,
                "age": patient.get("age"),
                "complaint": patient.get("complaint"),
                "final_triage": fields.get("final_triage"),
                "triage_reason_code": fields.get("triage_reason_code"),
                "final_advice": fields.get("final_advice")
            }
            execution_log.append(log_entry)
            print(f"Row {row_id}: Logged triage ({log_entry['final_triage']})")

        else:
            error_msg = submit_res.get("error", "Unknown")
            alert_entry = {
                "patient_row": row_id,
                "alert": f"Missing data — {error_msg}. Patient skipped."
            }
            execution_log.append(alert_entry)
            print(f"Row {row_id}: Submission failed — {error_msg}")

    # Save log file
    with open('final_execution_log.json', 'w') as f:
        json.dump(execution_log, f, indent=2)

    print(f"\nSync complete, {len(execution_log)} entries saved to 'final_execution_log.json'")


if __name__ == "__main__":
    main()