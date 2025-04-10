#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv
import PyPDF2
from groq import Groq
from pinecone import Pinecone
from pinecone import ServerlessSpec
import openai
from tqdm import tqdm

# Load environment variables from a .env file
load_dotenv()

# Instantiate the Groq client using your API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Please set your GROQ_API_KEY in the environment.")
    sys.exit(1)
groq_client = Groq(api_key=groq_api_key)
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text, chunk_size=1200, overlap=120):
    """Split text into overlapping chunks to preserve context."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    """Generate an embedding for the given text using the SentenceTransformer model."""
    response = openai.Embedding.create(
        input=text,
        engine="text-embedding-3-small"
    )
    # Return the embedding directly instead of a list of embeddings.
    return response['data'][0]['embedding']


def upsert_to_pinecone(index, chunks, batch_size=50):
    """Generate embeddings for each text chunk and upsert them into the Pinecone index in batches."""
    vectors = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        embedding = get_embedding(chunk)
        vector = {
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        }
        vectors.append(vector)
        
    if vectors:
        # Upsert in batches to avoid exceeding the payload size limit
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        print(f"Upserted {len(vectors)} vectors into Pinecone.")
    else:
        print("No vectors to upsert.")



def query_pinecone(index, query, top_k=5):
    """Query the Pinecone index for the most similar text chunks to the given query."""
    #print(query)
    query_embedding = get_embedding(query)
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result

def generate_answer_with_groq(context, question):
    """
    Use the Groq API to generate an answer based on the context and question.
    The prompt is constructed by concatenating context with the query.
    """
    print(context)
    print("-" * 10)
    prompt = (
        f"Do not use your pretraining just answer the questions based on the information provided"
        f"Answer the question using the context below:\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
        f"Cite the page number of the information in the answer, if possible.\n\n"
    )
    completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000
    )
    # Adjust the extraction of the answer based on the actual response structure.
    answer = completion.choices[0].message.content
    return answer

def main(pdf_path):
    # Initialize Pinecone with your API key and environment
    name = "mchip-chw"
    objectPine = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    index_name = name
    index = objectPine.Index(index_name)

    """
    # Extract text from PDF and split into manageable chunks
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    print(f"Extracted text split into {len(chunks)} chunks.")

    # Upsert the text chunks (with embeddings) into Pinecone
    upsert_to_pinecone(index, chunks)

    # Begin an interactive chat loop
    print("Indexing complete. You can now chat. Type 'exit' to quit.")
    """
    
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        # Query Pinecone for related chunks
        query_results = query_pinecone(index, user_query)
        #print(f"Query results: {query_results}")
        # Combine the retrieved chunks into a single context string
        context = "\n".join(match["metadata"]["text"] for match in query_results.get("matches", []))
        # Generate an answer using the Groq API
        answer = generate_answer_with_groq(context, user_query)
        print("Answer:", answer)
        print("-" * 50)

if __name__ == "__main__":
    pdf_path = "MCHIP_CHW-Ref-Guide.pdf"
    main(pdf_path)
