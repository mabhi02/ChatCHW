#!/usr/bin/env python
import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve your Pinecone API key from environment variables
API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    print("Please set the PINECONE_API_KEY in your environment.")
    exit(1)

# Initialize Pinecone client using only the API key
pinecone = Pinecone(api_key=API_KEY)

# Set the index name to check
INDEX_NAME = "where-there-is-no-doctor"

# Get list of indexes
indexes_response = pinecone.list_indexes()
print("Indexes response:")
print(indexes_response)

# Extract the index names from the response
index_names = [item["name"] for item in indexes_response.get("indexes", [])]

if INDEX_NAME in index_names:
    print(f"\nIndex '{INDEX_NAME}' exists.")

    # Get index configuration details
    index_description = pinecone.describe_index(INDEX_NAME)
    print("\nIndex description:")
    print(index_description)

    # Check and display the embedding dimension (if available)
    if "dimension" in index_description:
        dimension = index_description["dimension"]
        print(f"\nEmbedding size (dimension): {dimension}")
    else:
        print("\nCould not determine the embedding dimension from the index description.")

    # Connect to the index to retrieve stats
    index = pinecone.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print("\nIndex stats:")
    print(stats)

    # Calculate the total number of vectors across all namespaces
    total_vectors = 0
    for ns_stats in stats.get("namespaces", {}).values():
        total_vectors += ns_stats.get("vector_count", 0)
    print(f"\nTotal number of vectors in the index: {total_vectors}")

    if total_vectors > 0:
        print("The index is populated.")
    else:
        print("The index is empty.")
else:
    print(f"\nIndex '{INDEX_NAME}' does not exist.")
