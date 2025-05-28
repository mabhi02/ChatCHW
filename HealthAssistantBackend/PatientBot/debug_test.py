import requests
import json

def test_patient_bot():
    base_url = "http://localhost:5003"
    
    print("=== Testing PatientBot API ===")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Load a specific patient case
    print("\n=== Loading Patient Case 0 ===")
    try:
        response = requests.post(f"{base_url}/api/patient/load", 
                               json={"case_id": 0})
        print(f"Load patient status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Patient data: {json.dumps(data, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Load patient failed: {e}")
        return
    
    # Get patient info
    print("\n=== Getting Patient Info ===")
    try:
        response = requests.get(f"{base_url}/api/patient/info")
        print(f"Patient info status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Patient info: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Get patient info failed: {e}")
    
    # Test some questions
    questions = [
        "How old are you?",
        "What is your sex?", 
        "What is your complaint?",
        "Do you have a cough?",
        "What is your temperature?",
        "Do you have difficulty breathing?",
        "Do you have a runny nose?"
    ]
    
    print("\n=== Testing Questions ===")
    for question in questions:
        try:
            response = requests.post(f"{base_url}/api/patient/ask",
                                   json={"question": question})
            if response.status_code == 200:
                data = response.json()
                print(f"Q: {question}")
                print(f"A: {data.get('response', 'No response')}")
                print()
            else:
                print(f"Error asking '{question}': {response.text}")
        except Exception as e:
            print(f"Question failed '{question}': {e}")

if __name__ == "__main__":
    test_patient_bot() 