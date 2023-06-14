# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 120
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5


# Settings for OpenAI NLP models. Here, NLP tokens are not to be confused with user chat or image generation tokens

NLP_MODEL_NAME = "gpt-4-0613"
NLP_MODEL_MAX_TOKENS = 8000
NLP_MODEL_REPLY_MAX_TOKENS = 2000
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]


# Settings for vector store

CHROMA_DB_DIR = "_chroma_db"
VECTOR_N_RESULTS = 4  # The N nearest documents to retrieve based on the query, should be in line with NLP_MODEL_MAX_TOKENS size
