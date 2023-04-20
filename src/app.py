# App to load the vector database and let users to ask questions from it
import os
import time
import base64
import tarfile
import aiohttp
import asyncio
import argparse
import requests
import traceback
import configparser
from utils import *
from PIL import Image
import streamlit as st
from app_config import *
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--site", type=str, default=None, help="The site to load (section name in cfg file)")
parser.add_argument("--config", type=str, help="Path to configuration file", default="cfg/default.cfg")

# Sanity check inputs

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    print(f"Error parsing command line arguments: {traceback.format_exc()}")
    os._exit(e.code)

config_fn = os.path.join(FILE_ROOT, args.config)
if not os.path.exists(config_fn):
    st.error(f"Config file not found: {config_fn}")
    st.stop()


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
                print(f"Downloading vector database from {CHROMA_DB_URL}...")
                for i in range(N_RETRIES):
                    try:
                        r = requests.get(CHROMA_DB_URL, allow_redirects=True, timeout=TIMEOUT)
                        if r.status_code != 200:
                            raise Exception(f"HTTP error {r.status_code}: {r.text}")
                        with open(tarball_fn, "wb") as f:
                            f.write(r.content)
                        print(f"Saved vector database to {tarball_fn}")
                        time.sleep(COOLDOWN)
                        break
                    except:
                        if i == N_RETRIES - 1:
                            raise
                        print(f"Error downloading vector database: {traceback.format_exc()}")
                        print(f"Retrying in {COOLDOWN * BACKOFF ** i} seconds...")
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
    args.site = query_params["site"][0].lower()

if args.site is None:
    st.error("No site specified!")
    st.stop()

# Load the config file

config = get_config(config_fn)

try:
    section = config[args.site]
    INITIAL_PROMPT = section["initial_prompt"]
    ICON_FN = section["icon_fn"]
    BROWSER_TITLE = section["browser_title"]
    MAIN_TITLE = section["main_title"]
    SUBHEADER = section["subheader"]
    USER_PROMPT = section["user_prompt"]
    FOOTER_HTML = section["footer_html"]
    CHROMA_DB_URL = section["chroma_db_url"]
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
    contents: str = "",
    align: str = "left"
) -> str:
    # Formats the message in an chat fashion (user right, reply left)
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    file_path = os.path.join(FILE_ROOT, "assets", ICON_FN)
    src = f"data:image/gif;base64,{get_local_img(file_path)}"
    if align == "right":
        div_class = "human-line"
        color = "rgb(165, 239, 127)"
        file_path = os.path.join(FILE_ROOT, "assets", "user_icon.png")
        src = f"data:image/gif;base64,{get_local_img(file_path)}"
    icon_code = f"<img class='chat-icon' src='{src}' width=32 height=32 alt='avatar'>"
    formatted_contents = f"""
    <div class="{div_class}">
        {icon_code}
        <div class="chat-bubble" style="background: {color};">
        &#8203;{contents}
        </div>
    </div>
    """
    return formatted_contents


async def main(human_prompt: str) -> dict:
    res = {'status': 0, 'message': "Success"}
    try:

        # Strip the prompt of any potentially harmful html/js injections
        human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;")

        if len(human_prompt) < 20:
            res['status'] = 1
            res['message'] = "Please enter a longer prompt."
            return res

        # Update chat log
        st.session_state.LOG.append(f"Human: {human_prompt}")

        # Clear the input box after human_prompt is used
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            line = st.session_state.LOG[-1]
            contents = line.split("Human: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

            reply_box = st.empty()
            reply_box.markdown(get_chat_message(), unsafe_allow_html=True)

            # This is one of those small three-dot animations to indicate the bot is "writing"
            writing_animation = st.empty()
            file_path = os.path.join(FILE_ROOT, "assets", "loading.gif")
            writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

            # Perform vector-store lookup of the human prompt
            if len(st.session_state.LOG) > 2:
                human_prompt = st.session_state.LOG[-2].split("AI: ", 1)[1] + "  \n" + human_prompt
            docs = vector_db.similarity_search(human_prompt)

            if DEBUG:
                with st.sidebar:
                    st.subheader("Prompt")
                    st.markdown(human_prompt)
                    st.subheader("Reference materials")
                    st.json(docs, expanded=False)

            messages = [
                {'role': "user", 'content': INITIAL_PROMPT}
            ] + [
                {'role': "user", 'content': f"Datapoint: {x.page_content}\n\n({x.metadata['source'].rstrip('/')})"} for x in docs
            ] + [
                {'role': "user", 'content': human_prompt}
            ]

            async with aiohttp.ClientSession() as httpclient:
                # Call the OpenAI ChatGPT API for final result
                chatbot_res = await get_chatbot_reply_data_async(
                    httpclient,
                    messages,
                    os.getenv("OPENAI_API_KEY")
                )

                if DEBUG:
                    with st.sidebar:
                        st.subheader("chatbot_res")
                        st.json(chatbot_res, expanded=False)

                if chatbot_res['status'] != 0:
                    res['status'] = chatbot_res['status']
                    res['message'] = chatbot_res['message']
                    return res

            reply_text = chatbot_res['data']

            if reply_text.startswith("AI: "):
                reply_text = reply_text.split("AI: ", 1)[1]

            # Render the reply as chat reply
            reply_box.markdown(get_chat_message(reply_text), unsafe_allow_html=True)

            # Clear the writing animation
            writing_animation.empty()

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

# Load the vector database
persist_directory = os.path.join(FILE_ROOT, CHROMA_DB_DIR, args.site.replace(".", "_"))
with st.spinner("Loading vector database..."):
    vector_db = get_vector_db(persist_directory)

with footer:
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

# Initialize/maintain a chat log so we can keep tabs on previous Q&As

if "LOG" not in st.session_state:
    st.session_state.LOG = []

# Render chat history so far
with chat_box:
    for line in st.session_state.LOG:
        # For AI response
        if line.startswith("AI: "):
            contents = line.split("AI: ")[1]
            st.markdown(get_chat_message(contents), unsafe_allow_html=True)

        # For human prompts
        if line.startswith("Human: "):
            contents = line.split("Human: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

# Define an input box for human prompts
with prompt_box:
    human_prompt = st.text_input(USER_PROMPT, value="", key=f"text_input_{len(st.session_state.LOG)}")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:
    run_res = asyncio.run(main(human_prompt))
    if run_res['status'] == 0 and not DEBUG:
        st.experimental_rerun()

    else:
        if run_res['status'] != 0:
            if run_res['status'] == 1:
                st.warning(run_res['message'])
            else:
                st.error(run_res['message'])
        with prompt_box:
            if st.button("Show text input field"):
                st.experimental_rerun()
