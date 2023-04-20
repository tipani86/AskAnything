# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 60
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5


# Settings for OpenAI NLP models. Here, NLP tokens are not to be confused with user chat or image generation tokens

INITIAL_PROMPT = "You are a multi-lingual master summarizer of knowledge. Below are one or several datapoints relevant to a user's question. The actual question will be shown last. Use only the source datapoints presented below to answer the user question in the same language as was asked. Cite the source websites in your response. If the below references are not enough to answer the question, you must refuse to answer it, asking the user to reformulate the question instead."

NLP_MODEL_NAME = "gpt-3.5-turbo"
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1000
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]


# Settings for vector store

CHROMA_DB_DIR = "_chroma_db"
