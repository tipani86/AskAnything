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
from pathlib import Path
from app_config import *
from loguru import logger
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

FILE_ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=None, help="Path to configuration file")

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
def get_config(file_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_fn)
    return config


@st.cache_data(show_spinner=False)
def get_favicon(file_path: Path):
    # Load a byte image and return its favicon
    return Image.open(file_path)


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(FILE_ROOT / "style.css", "r") as f:
        return f"<style>{f.read()}</style>"
    

@st.cache_data(show_spinner=False)
def get_local_img(file_path: Path) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


@st.cache_resource(show_spinner=False)
def get_vector_db(file_path: Path) -> Chroma:
    if not file_path.is_dir():
        # Check whether the file of same name but .tar.gz extension exists, if so, extract it
        tarball_fn = file_path.with_suffix(".tar.gz")
        if not tarball_fn.is_file():
            # Download it from CHROMA_DB_URL
            try:
                logger.info(f"Downloading vector database from {CHROMA_DB_URL}...")
                if not tarball_fn.parent.exists():
                    tarball_fn.parent.mkdir(parents=True, exist_ok=True)
                for i in range(N_RETRIES):
                    try:
                        r = requests.get(CHROMA_DB_URL, allow_redirects=True, timeout=TIMEOUT)
                        if r.status_code != 200:
                            raise Exception(f"HTTP error {r.status_code}: {r.text}")
                        with open(tarball_fn, "wb") as f:
                            f.write(r.content)
                        logger.info(f"Saved vector database to {str(tarball_fn)}")
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
            tar.extractall(path=file_path.parent)

    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(file_path), embedding_function=embeddings)

# Get query parameters
query_params = st.experimental_get_query_params()
if "debug" in query_params and query_params["debug"][0].lower() == "true":
    st.session_state.DEBUG = True

if "DEBUG" in st.session_state and st.session_state.DEBUG:
    DEBUG = True

if "site" in query_params:
    args.config = Path("cfg") / f"""{query_params["site"][0].lower()}.cfg"""

if args.config is None:
    st.error("No site specified!")
    st.stop()

# Load the config file

config_fn = FILE_ROOT / args.config
if not config_fn.exists():
    st.error(f"Config file not found: {config_fn}")
    st.stop()

config_basename = config_fn.stem
config = get_config(config_fn)

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
favicon = get_favicon(FILE_ROOT / "assets" / ICON_FN)
st.set_page_config(
    page_title=BROWSER_TITLE,
    page_icon=favicon,
)


### OTHER FUNCTION DEFINITIONS ###


def get_chat_message(
    message: dict[str, str],
    loading: bool = False,
    loading_fp: str = FILE_ROOT / "assets" / "loading.gif",
    streaming: bool = False,
) -> None:
    # Formats the message in an basic chat fashion
    image_container, contents_container = st.columns([1, 11], gap="small")

    sources = ""

    role = message["role"]
    contents = message["content"]
    if role == "assistant":
        if "SOURCES: " in contents:
            contents, sources = contents.split("SOURCES: ", 1)
        file_path = FILE_ROOT / "assets" / ICON_FN
        src = f"data:image/gif;base64,{get_local_img(file_path)}"
    elif role == "user":
        file_path = FILE_ROOT / "assets" / "user_icon.png"
        image_data = get_local_img(file_path)
        src = f"data:image/gif;base64,{image_data}"       
    else:
        # Not a message that needs to be rendered (for example, system message)
        return

    with image_container:
        st.markdown(f"<img class='chat-icon' border=0 src='{src}' width=32 height=32>", unsafe_allow_html=True)

    with contents_container:
        st.markdown(contents)
        if len(sources) > 0:
            if streaming:
                pass
            else:
                sources = sources.split(",")
                if len(sources) > 0:
                    html = ""
                    for i, source in enumerate(sources):
                        if len(source.strip()) == 0:
                            continue
                        html += f"<a target='_BLANK' href='{source.strip()}'>[{i + 1}]</a>"
                    if len(html) > 0:
                        st.markdown(html, unsafe_allow_html=True)
        if loading:
            st.markdown(f"<img src='data:image/gif;base64,{get_local_img(loading_fp)}' width=30 height=10>", unsafe_allow_html=True)



async def main(human_prompt: str) -> tuple[int, str]:
    res_status = 0
    res_message = "Success"
    try:
        # Strip the prompt of any potentially harmful html/js injections
        human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;").strip()
        
        # Update chat log
        st.session_state.MESSAGES.append({
            "role": "user",
            "content": human_prompt
        })

        # Clear the input box after human_prompt is used
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            if len(st.session_state.MESSAGES) > 1:
                st.divider()
            
            message = st.session_state.MESSAGES[-1]
            get_chat_message(message)
            st.divider()

            reply_box = st.empty()
            with reply_box:
                get_chat_message({
                    "role": "assistant",
                    "content": ""
                }, loading=True)

            summary = ""
            if len(st.session_state.MESSAGES) > 1:
                # Summarize chat history so far
                history_str = "Please summarize the key topics and contents of the below conversation， in the same language as the conversation:\n\n"
                for message in st.session_state.MESSAGES[:-1]:
                    if message["role"] == "assistant":
                        history_str += "AI: "
                    elif message["role"] == "user":
                        history_str += "Human: "
                    history_str += message["content"] + "\n\n"
                history_str += "Summary:"

                if DEBUG:
                    with st.sidebar:
                        st.subheader("History_str")
                        st.markdown(history_str)
                        
                # Call GPT-3.5-Turbo model to summarize the conversation
                call_res = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": history_str}],
                    max_tokens=500,
                    temperature=0,
                    timeout=TIMEOUT,
                )
                summary = call_res["choices"][0]["message"]["content"].strip()

                # Create an adjusted human prompt that attaches the actual prompt text to the two ends of the summary.
                summary_prompt = f"{human_prompt}\n\n{summary}\n\n{human_prompt}"
            else:
                summary_prompt = human_prompt

            # Perform vector-store lookup of the human prompt
            docs = vector_db.similarity_search(summary_prompt)

            if DEBUG:
                with st.sidebar:
                    st.subheader("Prompt")
                    st.markdown(summary_prompt)
                    st.subheader("Reference materials")
                    st.json(docs, expanded=False)

            contents = []
            for doc in docs:
                processed_contents = f"Content: {doc.page_content}"
                if "source" in doc.metadata:
                    processed_contents += f"\n\nSource: {doc.metadata['source'].rstrip('/')}"
                elif "url" in doc.metadata:
                    processed_contents += f"\n\nSource: {doc.metadata['url'].rstrip('/')}"
                contents.append(processed_contents)
            contents = "\n##\n".join(contents)

            prompt = f"""{summary}
#### Instructions ####
{INITIAL_PROMPT}
#### Documents ####
{contents}
#### Question ####
{summary_prompt}"""

            messages = [
                {"role": "system", "content": prompt}
            ]

            if DEBUG:
                with st.sidebar:
                    st.subheader("Query")
                    st.json(messages, expanded=False)

            # Call the OpenAI ChatGPT API for final result
            reply_text = ""
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
                        get_chat_message({
                            "role": "assistant",
                            "content": reply_text
                        }, streaming=True)

            # Final fixing

            # Sanitizing output
            reply_text = reply_text.strip()
            if reply_text.startswith("AI: "):
                reply_text = reply_text.split("AI: ", 1)[1]

            st.session_state.MESSAGES.append({
                "role": "assistant",
                "content": reply_text
            })

    except:
        res_status = 2
        res_message = traceback.format_exc()

    return res_status, res_message

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
persist_directory = FILE_ROOT / CHROMA_DB_DIR / config_basename.replace(".", "_")
with st.spinner("Loading vector database..."):
    vector_db = get_vector_db(persist_directory)

with footer:
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
    st.write("")
    st.info(
        f"Note: This app uses OpenAI's GPT-4 under the hood. The service may sometimes be busy, so please wait patiently if the reply doesn't begin immediately.",
        icon="ℹ️")

# Initialize/maintain a chat log so we can keep tabs on previous Q&As

if "MESSAGES" not in st.session_state:
    st.session_state.MESSAGES = []

# Render chat history so far
with chat_box:
    for i, message in enumerate(st.session_state.MESSAGES):
        get_chat_message(message)
        # Add a divider between messages unless it's the last message
        if i < len(st.session_state.MESSAGES) - 1:
            st.divider()

# Define an input box for human prompts
with prompt_box:
    with st.form(key="Text input", clear_on_submit=True):
        human_prompt = st.text_input(USER_PROMPT, value="", placeholder=USER_PROMPT, label_visibility="collapsed", key=f"text_input_{len(st.session_state.MESSAGES)}")
        submitted = st.form_submit_button(label="Send")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if submitted and len(human_prompt) > 0:
    status, message = asyncio.run(main(human_prompt))
    if status == 0 and not DEBUG:
        st.experimental_rerun()
    else:
        if status != 0:
            st.error(message)
        with prompt_box:
            if st.button("Show text input field"):
                st.experimental_rerun()

