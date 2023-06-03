# App to load the vector database and let users to ask questions from it
import os
import time
import openai
import base64
import tarfile
import asyncio
import argparse
import requests
import traceback
import configparser
from PIL import Image
import streamlit as st
from app_config import *
from loguru import logger
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Path to configuration file")

# Sanity check inputs

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    logger.error(f"Error parsing command line arguments: {traceback.format_exc()}")
    os._exit(e.code)


### CACHED FUNCTION DEFINITIONS ###

@st.cache_data(show_spinner=False)
def get_config(file_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


@st.cache_data(show_spinner=False)
def get_favicon(file_path: str):
    # Load a byte image and return its favicon
    return Image.open(file_path)


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(FILE_ROOT, "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"
    

@st.cache_data(show_spinner=False)
def get_local_img(file_path: str) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


@st.cache_resource(show_spinner=False)
def get_vector_db(file_path: str) -> Chroma:
    if not os.path.isdir(file_path):
        # Check whether the file of same name but .tar.gz extension exists, if so, extract it
        tarball_fn = file_path.rstrip("/") + ".tar.gz"
        if not os.path.isfile(tarball_fn):
            # Download it from CHROMA_DB_URL
            try:
                logger.info(f"Downloading vector database from {CHROMA_DB_URL}...")
                if not os.path.exists(os.path.dirname(tarball_fn)):
                    os.makedirs(os.path.dirname(tarball_fn), exist_ok=True)
                for i in range(N_RETRIES):
                    try:
                        r = requests.get(CHROMA_DB_URL, allow_redirects=True, timeout=TIMEOUT)
                        if r.status_code != 200:
                            raise Exception(f"HTTP error {r.status_code}: {r.text}")
                        with open(tarball_fn, "wb") as f:
                            f.write(r.content)
                        logger.info(f"Saved vector database to {tarball_fn}")
                        time.sleep(COOLDOWN)
                        break
                    except:
                        if i == N_RETRIES - 1:
                            raise
                        logger.warning(f"Error downloading vector database: {traceback.format_exc()}")
                        logger.info(f"Retrying in {COOLDOWN * BACKOFF ** i} seconds...")
                        time.sleep(COOLDOWN * BACKOFF ** i)
            except:
                st.error(f"Error downloading vector database: {traceback.format_exc()}")
                st.stop()

        with tarfile.open(tarball_fn, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(file_path))

    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=file_path, embedding_function=embeddings)

# Get query parameters
query_params = st.experimental_get_query_params()
if "debug" in query_params and query_params["debug"][0].lower() == "true":
    st.session_state.DEBUG = True

if "DEBUG" in st.session_state and st.session_state.DEBUG:
    DEBUG = True

if "site" in query_params:
    args.config = f"""cfg/{query_params["site"][0].lower()}.cfg"""

if args.config is None:
    st.error("No site specified!")
    st.stop()

# Load the config file

config_fn = os.path.join(FILE_ROOT, args.config)
if not os.path.exists(config_fn):
    st.error(f"Config file not found: {config_fn}")
    st.stop()

config_basename, _ = os.path.splitext(os.path.basename(config_fn))
config = configparser.ConfigParser()
config.read(config_fn)

try:
    for section_name in config.sections():
        if section_name != "DEFAULT":
            section = config[section_name]
            INITIAL_PROMPT = section["initial_prompt"]
            ICON_FN = section["icon_fn"]
            BROWSER_TITLE = section["browser_title"]
            MAIN_TITLE = section["main_title"]
            SUBHEADER = section["subheader"]
            USER_PROMPT = section["user_prompt"]
            FOOTER_HTML = section.get("footer_html", "")
            CHROMA_DB_URL = section["chroma_db_url"]
            break
except:
    st.error(f"Error reading config file {config_fn}: {traceback.format_exc()}")
    st.stop()

# Initialize a Streamlit UI with custom title and favicon
favicon = get_favicon(os.path.join(FILE_ROOT, "assets", ICON_FN))
st.set_page_config(
    page_title=BROWSER_TITLE,
    page_icon=favicon,
)


### OTHER FUNCTION DEFINITIONS ###


def get_chat_message(
    line: str,
    loading: bool = False,
    loading_fp: str = os.path.join(FILE_ROOT, "assets", "loading.gif")
) -> None:
    # Formats the message in an basic chat fashion
    image_container, contents_container = st.columns([1, 11], gap="small")

    if line.startswith("AI: "):
        contents = line.split("AI: ", 1)[1]
        file_path = os.path.join(FILE_ROOT, "assets", ICON_FN)
        src = f"data:image/gif;base64,{get_local_img(file_path)}"
    elif line.startswith("Human: "):
        contents = line.split("Human: ", 1)[1]
        file_path = os.path.join(FILE_ROOT, "assets", "user_icon.png")
        image_data = get_local_img(file_path)
        src = f"data:image/gif;base64,{image_data}"
    else:
        st.error(f"Unknown message type: {line}")
        st.stop()

    with image_container:
        st.markdown(f"<img class='chat-icon' border=0 src='{src}' width=32 height=32>", unsafe_allow_html=True)

    with contents_container:
        st.markdown(contents)
        if loading:
            st.markdown(f"<img src='data:image/gif;base64,{get_local_img(loading_fp)}' width=30 height=10>", unsafe_allow_html=True)



async def main(human_prompt: str) -> dict:
    res = {'status': 0, 'message': "Success"}
    try:

        # Strip the prompt of any potentially harmful html/js injections
        human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;").strip()

        # Update chat log
        st.session_state.LOG.append(f"Human: {human_prompt}")

        # Clear the input box after human_prompt is used
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            if len(st.session_state.LOG) > 1:
                st.divider()
            
            line = st.session_state.LOG[-1]
            get_chat_message(line)
            st.divider()

            reply_box = st.empty()
            with reply_box:
                get_chat_message("AI: ", loading=True)

            # Perform vector-store lookup of the human prompt
            docs = vector_db.similarity_search(human_prompt, )

            if DEBUG:
                with st.sidebar:
                    st.subheader("Prompt")
                    st.markdown(human_prompt)
                    st.subheader("Reference materials")
                    st.json(docs, expanded=False)

            contents = "\n###\n".join([f"Content: {x.page_content}\n\nSource: {x.metadata['source'].rstrip('/')}" for x in docs])

            prompt = f"""
            {INITIAL_PROMPT}
            #### Contents ####
            {contents}
            #### Question ####
            {human_prompt}
            """

            messages = [
                {'role': "user", 'content': prompt}
            ]

            if DEBUG:
                with st.sidebar:
                    st.subheader("Query")
                    st.json(messages, expanded=False)

            # Call the OpenAI ChatGPT API for final result
            reply_text = "AI: "
            async for chunk in await openai.ChatCompletion.acreate(
                model=NLP_MODEL_NAME,
                messages=messages,
                max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                temperature=0,
                stop=NLP_MODEL_STOP_WORDS,
                stream=True,
                timeout=TIMEOUT,
            ):
                content = chunk["choices"][0].get("delta", {}).get("content", None)
                if content is not None:
                    reply_text += content

                    # Continuously render the reply as it comes in
                    with reply_box:
                        get_chat_message(reply_text)

            # Final fixing

            # Sanitizing output
            reply_text = reply_text.strip()
            if reply_text.startswith("AI: "):
                reply_text = reply_text.split("AI: ", 1)[1]

            sources = []
            if "SOURCES: " in reply_text:
                reply_text, sources = reply_text.split("SOURCES: ", 1)
                sources = sources.split(",")
            
            if len(sources) > 0:
                for i, source in enumerate(sources):
                    if len(source.strip()) == 0:
                        continue
                    html = f"<a target='_BLANK' href='{source.strip()}'>[{i + 1}]</a>"
                    reply_text += f" {html}"

            # Update the chat log
            st.session_state.LOG.append(f"AI: {reply_text}")

    except:
        res['status'] = 2
        res['message'] = traceback.format_exc()

    return res

### MAIN APP STARTS HERE ###


# Define main layout
st.title(MAIN_TITLE)
st.subheader(SUBHEADER)
st.subheader("")
chat_box = st.container()
st.write("")
prompt_box = st.empty()
footer = st.container()

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# # Load JS code
# components.html(get_js(), height=0, width=0)

# Load the vector database
persist_directory = os.path.join(FILE_ROOT, CHROMA_DB_DIR, config_basename.replace(".", "_"))
with st.spinner("Loading vector database..."):
    vector_db = get_vector_db(persist_directory)

with footer:
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.write("")
    st.info(
        f"Note: This app uses OpenAI's GPT-4 under the hood. The service may sometimes be busy, so please wait patiently if the reply doesn't begin immediately.",
        icon="ℹ️")

# Initialize/maintain a chat log so we can keep tabs on previous Q&As

if "LOG" not in st.session_state:
    st.session_state.LOG = []

# Render chat history so far
with chat_box:
    for i, line in enumerate(st.session_state.LOG):
        get_chat_message(line)
        # Add a divider between messages unless it's the last message
        if i < len(st.session_state.LOG) - 1:
            st.divider()

# Define an input box for human prompts
with prompt_box:
    with st.form(key="Text input", clear_on_submit=True):
        human_prompt = st.text_input(USER_PROMPT, value="", placeholder=USER_PROMPT, label_visibility="collapsed", key=f"text_input_{len(st.session_state.LOG)}")
        submitted = st.form_submit_button(label="Send")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if submitted and len(human_prompt) > 0:
    run_res = asyncio.run(main(human_prompt))
    if run_res['status'] == 0 and not DEBUG:
        st.experimental_rerun()
    else:
        if run_res['status'] != 0:
            st.error(run_res['message'])
        with prompt_box:
            if st.button("Show text input field"):
                st.experimental_rerun()

