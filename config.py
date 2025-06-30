"""
Configuration settings for RAG Rules Engine
"""

import os

# App settings
APP_TITLE = "RAG Rules Engine"
DEBUG = True
HOST = "127.0.0.1"
PORT = 8050

# Database settings (placeholder)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///rules.db")

# Embedding model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQUENCE_LENGTH = 512

# Search settings
DEFAULT_SEARCH_LIMIT = 20
SIMILARITY_THRESHOLD = 0.7

# Chat settings
MAX_CHAT_HISTORY = 50
DEFAULT_RESPONSE_MAX_LENGTH = 500

# File paths
DATA_DIR = "data"
MODELS_DIR = "models"
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)