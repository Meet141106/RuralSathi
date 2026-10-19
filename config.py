import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
RERANK_TOP_K = 3
VECTOR_STORE_PATH = "vector_store/ruralsathi.index"
METADATA_PATH = "vector_store/metadata.pkl"
TEMPERATURE = 0
MAX_TOKENS = 600
