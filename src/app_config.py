# Debug switch
DEBUG = False


# Generic internet settings
TIMEOUT = 120
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5


# Settings for OpenAI NLP models. Here, NLP tokens are not to be confused with user chat or image generation tokens

NLP_MODEL_NAME = "gpt-4"
NLP_MODEL_MAX_TOKENS = 16000
NLP_MODEL_FUNCTIONS_TOKENS = 1000
NLP_MODEL_REPLY_MAX_TOKENS = 2000

# Basic prompt settings (more customized need to be imported through config files)

PRE_SUMMARY_PROMPT = "The above is the conversation so far between you, the AI assistant, and a human user. Please summarize the topics discussed for your own reference. Remember, do not write a direct reply to the user."

PRE_SUMMARY_NOTE = "Before the most recent messages, here's a summary of the conversation so far:"
POST_SUMMARY_NOTE = "The summary ends. And here are the most recent two messages from the conversation. You should generate the next response based on the conversation so far."


# Settings for vector store

CHROMA_DB_DIR = "_chroma_db"
VECTOR_N_RESULTS = 6  # The N nearest documents to retrieve based on the query, should be in line with NLP_MODEL_MAX_TOKENS size
