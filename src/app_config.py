# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 60
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5


# Settings for OpenAI NLP models. Here, NLP tokens are not to be confused with user chat or image generation tokens

INITIAL_PROMPT = "You are a multi-lingual master librarian and the summarizer of knowledge. Below is one or several datapoints relevant to a user's question. The actual question will be shown last. Your job is to answer the question, in the same language, using only the datapoints presented below. Always note the source website when you cite or adapt a certain datapoint in your response. You may not consult any extra sources or your inherent memory when generating the results."

NLP_MODEL_NAME = "gpt-4"
NLP_MODEL_MAX_TOKENS = 8000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]


# Settings for vector store

CHROMA_DB_DIR = "_chroma_db"
