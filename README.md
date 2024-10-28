# ChatCHW (Community Health Worker Assistant)

An AI-powered health consultation application designed to support Community Health Workers (CHWs) in providing preliminary health assessments and guidance. Built with React Native (Expo) and powered by advanced language models, ChatCHW streamlines the patient consultation process through intelligent conversation flows and evidence-based recommendations.

## 📑 Table of Contents

1. [Features](#-features)
2. [Technical Stack](#️-technical-stack)
3. [ML Architecture & Data Flow](#-ml-architecture--data-flow)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
6. [Running the Application](#️-running-the-application)
7. [Application Flow](#-application-flow)
8. [System Architecture](#-system-architecture)
9. [API Endpoints](#-api-endpoints)
10. [Testing](#-testing)
11. [Security & Privacy](#-security--privacy)
12. [Future Development](#-future-development)
13. [License](#-license)
14. [Authors](#-repo-authors)
15. [Acknowledgments](#-acknowledgments)
16. [Disclaimer](#️-disclaimer)

## 🌟 Features

### Smart Health Screening
- **Comprehensive Patient Information Collection**
  - Demographic data gathering
  - Medical history documentation
  - Current symptoms assessment
  - Caregiver information recording
  - Accompaniment status tracking

### AI-Powered Chat Interface
- **Intelligent Conversation System**
  - Dynamic follow-up question generation
  - Evidence-based response formulation
  - Context-aware conversation management
  - Symptom analysis and tracking
  - Medical knowledge integration

### Multi-Stage Consultation Process
1. **Initial Screening**
   - Patient information collection
   - Primary symptom assessment
   - Risk factor identification

2. **Follow-up Questions**
   - Symptom clarification
   - Medical history exploration
   - Context gathering

3. **Test Recommendations**
   - Evidence-based test suggestions
   - Procedure explanations
   - Risk assessment

4. **Professional Advice**
   - Treatment recommendations
   - Lifestyle modifications
   - Follow-up scheduling

5. **Summary & Documentation**
   - Consultation summary
   - Care plan documentation
   - Reference materials

### User Interface
- **Dark Mode Optimization**
  - Eye-friendly design
  - High contrast elements
  - Accessible color scheme
  - Responsive layouts
  - Intuitive navigation

## 🛠️ Technical Stack

### Frontend Technologies
- **React Native (Expo)**
  - Cross-platform compatibility
  - Fast development cycle
  - Hot reloading support
  - Native functionality access

- **TypeScript**
  - Type safety
  - Enhanced development experience
  - Better code maintainability

- **Navigation & UI**
  - React Navigation for routing
  - Vector Icons for visual elements
  - Native Picker for form inputs

### Backend Infrastructure
- **Flask**
  - RESTful API implementation
  - Lightweight server architecture
  - Easy integration capabilities

- **Database**
  - MongoDB for conversation storage
  - Flexible document structure
  - Scalable data management

- **AI Integration**
  - Groq API for language processing
  - OpenAI for embedding generation
  - Pinecone for vector storage

## 🧠 ML Architecture & Data Flow

### Vector Database Implementation
#### Pinecone Integration
- Database name: "final-asha"
- Optimized for medical knowledge storage
- High-performance similarity search
- Scalable vector storage

#### Vector Operations
- Dimension: 1536-dimensional space
- Indexing: HNSW algorithm
- Search: Cosine similarity metrics
- Query optimization: Top-k retrieval

### Embedding Generation Process
#### OpenAI Integration
- Model: text-embedding-3-small
- Input processing: Text normalization
- Output: 1536-dimensional vectors
- Batch processing capabilities

### Language Model Pipeline
#### Groq LLM Configuration
- Model: llama3-70b-8192
- Context window: 8192 tokens
- Temperature: Dynamic based on task
- Response formatting: Task-specific

### Conversation Stage Management
#### Follow-up Questions Stage
```python
def generate_followup_question(main_symptom, screening_info, previous_questions, relevant_quote):
    prompt = f"""Generate a single, concise follow-up question about: "{main_symptom}"
    Consider this screening information: {screening_info}
    The question should be different from these previous questions: {previous_questions}
    Base your question on this relevant information: {relevant_quote['text']}
    Provide only the question, nothing else."""
    return groqCall(prompt)
```

#### Test Generation Stage
```python
def generate_test(main_symptom, screening_info, symptom_info, relevant_quote, previous_tests):
    prompt = f"""Suggest a simple test for: "{main_symptom}"
    Consider this screening information: {screening_info}
    And these symptoms: {symptom_info}
    The test should be different from these previous tests: {previous_tests}
    Base your suggestion on this relevant information: {relevant_quote['text']}
    Provide the test description and instructions."""
    return groqCall(prompt)
```

#### Advice Formation Stage
```python
def generate_advice(main_symptom, screening_info, symptom_info, test_results, relevant_quote):
    prompt = f"""Provide 3 pieces of advice for: "{main_symptom}"
    Consider this screening information: {screening_info}
    These symptoms: {symptom_info}
    And these test results: {test_results}
    Base your advice on this relevant information: {relevant_quote['text']}"""
    return groqCall(prompt)
```

## 📋 Prerequisites

### Development Environment
- Node.js (v14 or higher)
- Python 3.8+
- MongoDB
- Expo CLI (`npm install -g expo-cli`)

### API Keys Required
- OpenAI API key
- Groq API key
- Pinecone API key

## 🚀 Installation

### 1. Repository Setup
```bash
git clone https://github.com/yourusername/chatchw.git
cd chatchw
```

### 2. Frontend Installation
```bash
# Install dependencies
npm install

# Start the Expo development server
npx expo start
```
Then press 'w' to open the web version of the app.

### 3. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the backend directory:
```
OPENAI_API_KEY=
GROQ_API_KEY=
PINECONE_API_KEY=
```

## 🏃‍♂️ Running the Application

### Backend Server Launch
```bash
python app.py
```

### Frontend Development Server
```bash
npx expo start
```
Press 'w' to run in web browser.

## 🔄 Application Flow

### 1. Initial Screening Process
- Patient demographics collection
- Medical history recording
- Current symptoms assessment
- Risk factor identification

### 2. AI Chat Interface Operation
- Symptom analysis
- Follow-up question generation
- Context maintenance
- Response formulation

### 3. Test Recommendation System
- Evidence-based suggestions
- Procedural instructions
- Risk assessment integration

### 4. Professional Advice Generation
- Treatment recommendations
- Lifestyle modification suggestions
- Follow-up scheduling assistance

## 🎯 System Architecture

### Frontend Component Structure
```
src/
├── components/
│   ├── Questionnaire.tsx
│   ├── Chats.tsx
│   └── App.tsx
├── styles/
│   └── theme.ts
└── utils/
    └── api.ts
```

### Backend Organization
```
backend/
├── app.py
├── ml_functions.py
├── templates/
│   ├── logs_template.html
│   └── conversation_log_template.html
└── utils/
    └── database.py
```

### Data Flow Patterns
1. User Input Processing
```
User Input → Frontend → API → ML Pipeline → Response
```

2. Conversation Management
```
Chat History → MongoDB → Context Assembly → Response
```

3. Knowledge Integration
```
Symptoms → Embedding → Vector Search → Knowledge Retrieval
```

## 🔍 API Endpoints

### Frontend Routes
- `/` - Initial questionnaire screen
- `/chats` - Main chat interface

### Backend Endpoints
- `GET /initial_questions` - Retrieve screening questions
- `POST /chat` - Process chat messages and generate responses
- `GET /full_conversation/:id` - Retrieve complete conversation history
- `GET /log` - View conversation logs (admin)
- `GET /log/:id` - View specific conversation details (admin)

## 🔬 Testing

### Frontend Testing
```bash
# Run frontend tests
npm test
```

### Backend Testing
```bash
# Run backend tests
python -m pytest
```

## 🔒 Security & Privacy

### API Security Implementation
- Rate limiting
- Input validation
- Request sanitization
- Token authentication

### Data Protection
- Medical data encryption
- Secure storage practices
- Access control implementation
- Audit logging

### Compliance
- Healthcare data regulations
- Privacy standards
- Security best practices

## 🚀 Future Development

### Planned Features
1. Multi-language Support
   - Interface translation
   - Medical terminology localization
   - Cultural adaptation

2. Offline Capabilities
   - Local data storage
   - Sync mechanisms
   - Offline ML model support

3. Enhanced Analytics
   - Usage patterns
   - Outcome tracking
   - Performance metrics

4. Integration Capabilities
   - EHR system connection
   - Healthcare provider linking
   - Telemedicine features

## 📝 License

This project is licensed under the APACHE License - see the [LICENSE.md](LICENSE.md) file for details.

## 👥 Repo Authors

- Abhijith Varma Mudunuri - *Engineering* - [mabhi02](https://github.com/mabhi02)

## 🙏 Acknowledgments

- Groq for providing the language model API
- OpenAI for embedding capabilities
- Pinecone for vector database services
- The open-source community for various tools and libraries used

## ⚠️ Disclaimer

ChatCHW is designed to assist Community Health Workers and should not be used as a replacement for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.

---

For additional information, bug reports, or feature requests, please open an issue in the GitHub repository.