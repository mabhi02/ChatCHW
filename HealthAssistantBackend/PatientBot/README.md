# PatientBot Knowledge-Base API

A knowledge-base driven patient simulation API that uses **ONLY** the information from `data.csv` columns A-G to respond to questions. This ensures realistic, constrained patient responses for CHW training.

## 🎯 Key Features

- **Knowledge-Base Only**: Responses come strictly from CSV data (columns A-G)
- **No External Medical Knowledge**: No AI-generated medical advice or assumptions
- **Realistic Patient Simulation**: Based on 1000+ real patient cases
- **Constrained Responses**: Returns "other: Information not known" when data unavailable
- **CHW Training Ready**: Perfect for testing assessment workflows

## 📊 Data Structure (Columns A-G)

| Column | Field | Description |
|--------|-------|-------------|
| A | Age Category | Child/Adult/Elder |
| B | Age (years) | Specific age in years |
| C | Sex | Male/Female |
| D | Complaint | Primary symptoms (Cough, Diarrhea, Sore throat) |
| E | Duration | How long symptoms present |
| F | CHW Questions | Detailed symptom responses |
| G | Exam Findings | Physical examination results |

## 🚀 Quick Start

### 1. Start the API
```bash
cd HealthAssistantBackend/PatientBot
python API.py
```

### 2. Run Quick Test
```bash
python quick_test.py
```

### 3. Run Full Test Suite
```bash
python test_knowledge_bot.py
```

## 🌐 API Endpoints

### Health Check
```http
GET /health
```
Returns API status and number of loaded cases.

### Load Patient Case
```http
POST /api/patient/load
Content-Type: application/json

{
  "case_id": 0  // Optional, random if not provided
}
```

### Ask Patient Question
```http
POST /api/patient/ask
Content-Type: application/json

{
  "question": "How bad is your cough?"
}
```

### Get Patient Info
```http
GET /api/patient/info
```

### List Available Cases
```http
GET /api/patient/cases
```

### Demo Conversation
```http
POST /api/patient/demo
Content-Type: application/json

{
  "case_id": 1  // Optional
}
```

## 💬 Example Usage

```python
import requests

# Load a patient
response = requests.post("http://localhost:5003/api/patient/load", 
                        json={"case_id": 0})
patient = response.json()['patient']
print(f"Patient: {patient['age_years']} {patient['sex']} with {patient['primary_complaint']}")

# Ask questions
questions = [
    "What is wrong with you?",
    "How bad is your cough?", 
    "Do you have fever?",
    "What medications are you taking?"  # Will return "Information not known"
]

for question in questions:
    response = requests.post("http://localhost:5003/api/patient/ask",
                           json={"question": question})
    answer = response.json()['response']
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

## 🔍 Knowledge Base Constraints

### ✅ Available Information
- Demographics (age, sex)
- Primary complaint and duration
- Symptom details from CHW Questions
- Physical exam findings
- Specific measurements (temperature, respiratory rate, etc.)

### ❌ Unavailable Information
- Medical history
- Medications
- Allergies
- Social history
- Personal preferences
- Anything not in columns A-G

### 🤖 Response Behavior
- **Known data**: Extracted directly from CSV
- **Unknown data**: Returns "other: Information not known"
- **No assumptions**: Never generates medical advice or guesses

## 🧪 Testing

### Quick Test
```bash
python quick_test.py
```
Basic functionality verification.

### Comprehensive Test
```bash
python test_knowledge_bot.py
```
Full test suite including:
- API endpoint testing
- Knowledge base constraint validation
- Specific case scenario testing
- Response accuracy verification

## 🎯 Use Cases

### CHW Training
- Test assessment workflows
- Practice diagnostic questioning
- Validate knowledge extraction

### System Integration
- Integrate with workflow APIs
- Test with assessment systems
- Validate CHW protocols

### Research & Development
- Analyze questioning patterns
- Test diagnostic algorithms
- Validate assessment tools

## 📝 Example Responses

```
Q: What is wrong with you?
A: Cough

Q: How bad is your cough?
A: Mild

Q: Do you have fever?
A: Temperature is 37.7°C

Q: What medications are you taking?
A: other: Information not known

Q: Do you have difficulty breathing?
A: Yes

Q: What is your favorite color?
A: other: Information not known
```

## 🔧 Configuration

The API runs on port 5003 by default. To change:

```python
app.run(debug=True, port=YOUR_PORT, host='0.0.0.0')
```

## 📋 Requirements

- Python 3.7+
- Flask
- pandas
- requests (for testing)

## 🎉 Success Criteria

A working PatientBot should:
1. ✅ Load patient cases from CSV
2. ✅ Extract answers from columns A-G only
3. ✅ Return "Information not known" for unavailable data
4. ✅ Never generate external medical knowledge
5. ✅ Respond consistently based on loaded case data

## 🚨 Important Notes

- **No Medical Advice**: This is a simulation tool only
- **Knowledge-Base Constrained**: Responses limited to CSV data
- **Training Purpose**: Designed for CHW education and testing
- **No External APIs**: No calls to medical databases or AI services 