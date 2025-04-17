# NeonDB Implementation in ChatCHW

## Overview
NeonDB is a serverless PostgreSQL database service used in the ChatCHW application for persistence and data analysis. The system uses NeonDB to store conversation history, retrieval-augmented generation (RAG) chunks, session summaries, and matrix evaluations from the AI decision-making process.

## Connection Configuration

### Connection Details
- **Connection String Format**: `postgresql://username:password@hostname/database?sslmode=require`
- **Current Configuration**: The application uses the connection string stored in environment variables (`DATABASE_URL` or `NEON_DATABASE_URL`)
- **Connection Example**: `postgresql://neondb_owner:npg_y4KVp2gxAaHh@ep-shrill-cloud-a4to19jn-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require`

### Connection Establishment
```python
# Initialize NeonDB with the correct URL
neondb_url = os.environ.get('DATABASE_URL') or os.environ.get('NEON_DATABASE_URL')
db_conn = None
if psycopg2 and neondb_url:
    try:
        # Connect to the database
        db_conn = psycopg2.connect(neondb_url)
        db_conn.autocommit = True  # Use autocommit mode
        
        # Test connection with a simple query
        with db_conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                print("NeonDB connected successfully")
    except Exception as e:
        print(f"Error connecting to NeonDB: {e}")
```

## Database Schema

The application uses four main tables in NeonDB:

### 1. conversation_messages Table
```sql
CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    message_type TEXT NOT NULL,  -- 'user' or 'assistant'
    phase TEXT NOT NULL,         -- 'initial', 'followup', 'exam', or 'complete'
    content TEXT NOT NULL,       -- The actual message content
    metadata JSONB               -- Additional structured data
)
```

### 2. rag_chunks Table
```sql
CREATE TABLE IF NOT EXISTS rag_chunks (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    phase TEXT,                  -- Stage of conversation (followup_question, examination, etc.)
    chunk_type TEXT,             -- Type of chunk (retrieved, used_for_generation, etc.)
    chunk_id TEXT,               -- Identifier for the retrieved chunk
    source TEXT,                 -- Source document reference
    text TEXT,                   -- The actual content of the chunk
    relevance_score FLOAT        -- Similarity score from vector search
)
```

### 3. session_summaries Table
```sql
CREATE TABLE IF NOT EXISTS session_summaries (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    initial_complaint TEXT,      -- Initial description of the medical issue
    diagnosis TEXT,              -- Final diagnosis generated
    treatment TEXT,              -- Recommended treatment
    chunks_used INTEGER          -- Number of medical guide chunks used
)
```

### 4. matrix_evaluations Table
```sql
CREATE TABLE IF NOT EXISTS matrix_evaluations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE,
    phase TEXT,                  -- Stage of conversation
    question TEXT,               -- The question being evaluated
    confidence FLOAT,            -- Confidence score from MATRIX
    optimist_weight FLOAT,       -- Weight assigned to optimistic agent
    pessimist_weight FLOAT,      -- Weight assigned to pessimistic agent
    selected_agent TEXT          -- Which agent's recommendation was selected
)
```

## Data Flow

### Conversation Message Storage
1. When a user interacts with the chat system, messages are sent to the backend API
2. The backend processes these messages and saves them to NeonDB using `save_conversation_message()`
3. Both user messages and assistant responses are stored with appropriate metadata
4. This creates a complete, searchable history of all conversations

### Example flow:
```python
save_conversation_message(
    session_id=session_id,
    message_type='assistant',  # or 'user'
    phase='initial',           # or 'followup', 'exam', 'complete'
    content=output,            # the text message
    metadata={                 # structured additional data
        'question_type': question_type,
        'options': options 
    }
)
```

### RAG Chunks Storage
1. During generation of questions or examinations, the system retrieves relevant chunks from a medical guide
2. These chunks and their metadata are stored in the `rag_chunks` table using `save_session_chunks_to_neondb()`
3. Allows for analysis of which medical information was most relevant

### MATRIX Evaluation Storage
1. The MATRIX system evaluates similarity between questions/examinations
2. Results of these evaluations are stored in the `matrix_evaluations` table
3. Helps track the decision-making process of the AI system

### Session Summary Storage
1. At the end of a conversation, a summary is generated with diagnosis and treatment
2. This summary is stored in the `session_summaries` table
3. Provides quick access to key information without parsing the entire conversation

## Data Retrieval

The system provides an API endpoint to retrieve conversation history:

```python
@app.route('/api/conversation/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get the conversation history for a session"""
    try:
        if not db_conn:
            return jsonify({
                "status": "error",
                "message": "Database connection not available"
            })
        
        with db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get all messages for this session, ordered by timestamp
            query = """
            SELECT * FROM conversation_messages 
            WHERE session_id = %s 
            ORDER BY timestamp ASC
            """
            cursor.execute(query, (session_id,))
            messages = cursor.fetchall()
            
            # Convert PostgreSQL types to JSON-serializable types
            for message in messages:
                if 'timestamp' in message and message['timestamp']:
                    message['timestamp'] = message['timestamp'].isoformat()
            
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "message_count": len(messages),
                "messages": messages
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
```

## Frontend Integration

The frontend uses Prisma, an ORM (Object-Relational Mapper), to interact with the same NeonDB:

```javascript
// Prisma schema
datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model Conversation {
  id        String    @id @default(uuid())
  sessionId String    @unique
  messages  Message[]
  metadata  Json?
  createdAt DateTime  @default(now())
  updatedAt DateTime  @updatedAt
}

model Message {
  id             String       @id @default(uuid())
  type           String      
  content        String
  metadata       Json?
  conversation   Conversation @relation(fields: [conversationId], references: [id])
  conversationId String
  createdAt      DateTime    @default(now())
}
```

## Implementation Requirements for a New Website

To implement a similar NeonDB setup on a separate website, you'll need:

1. **NeonDB Account Setup**:
   - Sign up at neon.tech
   - Create a new project
   - Create a new database
   - Get your connection string

2. **Environment Configuration**:
   - Store the connection string in environment variables

3. **Database Connection Code**:
   - Implement similar connection code using psycopg2 (Python) or @neondatabase/serverless (JavaScript)

4. **Schema Creation**:
   - Execute similar CREATE TABLE statements to establish your schema

5. **Data Storage Functions**:
   - Implement functions for storing different types of data

6. **Data Retrieval Endpoints**:
   - Create API endpoints to access stored data

7. **Error Handling**:
   - Implement proper error handling for database operations

8. **Optional: Analytics Dashboard**:
   - Create a dashboard to visualize conversation data
   - Enable analysis of conversation patterns and outcomes

The current implementation provides a robust foundation for conversation storage, medical information retrieval tracking, and AI decision-making processes in a healthcare context. It can be adapted to other domains by modifying the schema to match your specific data requirements. 