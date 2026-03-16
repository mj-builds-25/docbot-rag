# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Qdrant
QDRANT_PATH     = "./src/qdrant_data"
COLLECTION_NAME = "hr_docs"
VECTOR_DIM      = 384

# Chunking
CHUNK_SIZE = 100  # words per chunk

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama3-8b-8192"

# Document to ingest
DOC_URL = "https://raw.githubusercontent.com/tnahddisttud/sample-doc/refs/heads/main/atliqai_hr_policies.txt"