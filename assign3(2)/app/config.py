# app/config.py

import os

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# ChromaDB settings
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_collection"

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File paths
PDF_STORAGE_PATH = "temp_pdfs"