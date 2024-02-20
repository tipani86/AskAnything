# App to load the vector database and let users to ask questions from it
import os
import time
import json
import openai
import tarfile
import asyncio
import argparse
import requests
import traceback
import configparser
from utils import *
from PIL import Image
from typing import Any
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
    config.read(file_path)
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

    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002") if api_type == "azure" else OpenAIEmbeddings()
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
    args.config = Path("cfg") / "uk_frs.cfg"
    # st.error("No site specified!")
    # st.stop()

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

# Sanity check on environmental variables
env_keys = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_TYPE"]
env_var_errors = []
for key in env_keys:
    if key not in os.environ:
        env_var_errors.append(key)

api_type = os.environ.get("OPENAI_API_TYPE")
    
if len(env_var_errors) > 0:
    logger.error(f"Please set the following environment variables: {env_var_errors}")
    os._exit(2)

# Initialize a Streamlit UI with custom title and favicon
favicon = get_favicon(FILE_ROOT / "assets" / ICON_FN)
st.set_page_config(
    page_title=BROWSER_TITLE,
    page_icon=favicon,
)


### OTHER FUNCTION DEFINITIONS ###

async def get_model_reply_async(
    messages: list,
    model_name: str,
    max_tokens: int,
    streaming: bool = False,
    temperature: float = 0.0,
    custom_init_prompt: str | None = None,
    container: Any | None = None,   # Should be an st.empty() object if streaming is True
    function_call: str | dict[str, Any] = "none",
    functions: dict[str, Any] | None = None,
) -> tuple[int, str, dict[str, Any] | None]:
    res_status = 0
    res_message = "Success"
    res_data = None

    if functions is None:
        functions = {"available_funs": [], "api_in": [{"name": "_default", "parameters": {"type": "object", "properties": {}}}]}

    try:
        # If custom_init_prompt is provided, use it as the initial prompt
        if custom_init_prompt is not None:
            if messages[0]["role"] == "system":
                messages[0]["content"] = custom_init_prompt
            else:
                # Put a new message before others
                messages.insert(0, {"role": "system", "content": custom_init_prompt})

        if "DEBUG" in st.session_state and st.session_state.DEBUG:
            with st.sidebar:
                st.write("Input messages")
                st.json(messages, expanded=False)
        
        if not streaming:
            response = await openai.ChatCompletion.acreate(
                model=model_name,
                deployment_id=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=TIMEOUT,
                function_call=function_call,
                functions=functions["api_in"],
            )
            first_choice_message = response["choices"][0]["message"]
            if first_choice_message["content"] is not None:
                reply_text = first_choice_message["content"].strip()
                function_call = None
            else:
                reply_text = ""
                function_call = first_choice_message["function_call"]
        else:
            # For streaming, we need to loop through the response generator
            reply_text = ""
            function_name = ""
            function_args = ""
            async for chunk in await openai.ChatCompletion.acreate(
                model=model_name,
                deployment_id=model_name,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                timeout=TIMEOUT,
                function_call=function_call,
                functions=functions["api_in"],
            ):
                try:
                    delta = chunk["choices"][0].get("delta", {})
                except: # Known bug in Azure OpenAI API where the first streamed chunk is empty
                    continue
                content = delta.get("content", None)
                function_call = delta.get("function_call", None)
                if function_call is not None:
                    function_name += function_call.get("name", "")
                    function_args += function_call.get("arguments", "")
                if content is not None:
                    reply_text += content

                    # Sanitize output
                    if reply_text.startswith("AI: "):
                        reply_text = reply_text.split("AI: ", 1)[1]

                    message = {"role": "assistant", "content": reply_text}

                    # Continuously write the response in Streamlit, if container is provided (here the container should be an st.empty() object)
                    if container is not None:
                        with container:
                            get_chat_message(-1, message, streaming=True)

            # Collect full function call
            if function_name != "" and function_args != "":
                function_call = {"name": function_name, "arguments": function_args}
            else:
                function_call = None

        # Process final output

        # Check whether the model wants to call a function and call it, if appropriate
        if function_call is not None:

            if "DEBUG" in st.session_state and st.session_state.DEBUG:
                with st.sidebar:
                    st.write("Function call response")
                    st.json(function_call, expanded=False)

            # Read the function call from model response and execute it (if appropriate)
            available_funs = functions["available_funs"]
            fun_name = function_call.get("name", None)
            if fun_name is not None and fun_name and fun_name in available_funs:
                function = available_funs[fun_name]
            else:
                function = None
            fun_args = function_call.get("arguments", None)
            if fun_args is not None and isinstance(fun_args, str):
                fun_args = json.loads(fun_args)
            if function is not None:
                with st.status(f"Called function `{fun_name}`"):
                    st.json(fun_args, expanded=True)
                    fun_res = function(fun_args)
            else:
                fun_res = ["Error, no function specified"]

            out_messages = [{"role": "function", "name": fun_name, "content": one_fun_res} for one_fun_res in fun_res]

        else:   # Not a function call, return normal message

            # Sanitize
            if reply_text.startswith("AI: "):
                reply_text = reply_text.split("AI: ", 1)[1]

            out_messages = [{"role": "assistant", "content": reply_text}]

            # Write the response to Streamlit if container is provided
            if container is not None:
                with container:
                    get_chat_message(-1, out_messages[0])

        res_data = {
            "messages": out_messages,
            "function_call": function_call,
        }
    except:
        return 2, traceback.format_exc(), None

    return res_status, res_message, res_data


# Check whether tokenized memory so far + max reply length exceeds the max possible tokens for the model.
# If so, summarize the middle part of the memory using the model itself, re-generate the memory.
async def generate_prompt_from_memory_async(
    messages: list,
    model_name: str,
    max_tokens: int,
    functions_tokens: int,
    reply_max_tokens: int,
) -> dict:
    res_status = 0
    res_message = "Success"
    res_data = None

    n_tokens_memory = num_tokens_from_messages(messages, model_name)
    
    if "DEBUG" in st.session_state and st.session_state.DEBUG:
        with st.sidebar:
            st.write(f"n_tokens_memory: {n_tokens_memory}")

    buffer = 1.2    # Extend a 20% buffer in the clause to account for additional prompts etc.

    if n_tokens_memory * buffer + functions_tokens + reply_max_tokens > max_tokens:
        # Strategy: We keep the and last three items
        # (last AI message and human's reply and possibly a function response) intact,
        # and summarize all that comes before that.
        summarizable_memory = messages[:-3]

        # We write a new prompt asking the model to summarize this middle part
        summarizable_memory += [{'role': "system", 'content': PRE_SUMMARY_PROMPT}]
        n_tokens_summarizable = num_tokens_from_messages(summarizable_memory, model_name)

        # Check whether the summarizable tokens + 75% of the reply length exceeds the max possible tokens.
        # If so, adjust down to 50% of the reply length and try again, lastly if even 25% of the reply tokens still exceed, call an error.
        for ratio in [0.75, 0.5, 0.25]:
            if n_tokens_summarizable + int(reply_max_tokens * ratio) <= max_tokens:
                # Call the OpenAI API
                summary_status, summary_message, summary_data = await get_model_reply_async(
                    messages=summarizable_memory,
                    model_name=model_name,
                    max_tokens=int(reply_max_tokens * ratio),
                )
                if summary_status != 0:
                    return summary_status, summary_message, summary_data

                # Re-build memory so it consists of a note that a summary follows, then
                # the actual summary, and a second note that the last two conversation items follow,
                # then the last two items from the original memory
                new_memory = [{'role': "user", "content": text} for text in [PRE_SUMMARY_NOTE, summary_data["messages"][0]["content"], POST_SUMMARY_NOTE]] + messages[-3:]

                if "DEBUG" in st.session_state and st.session_state.DEBUG:
                    with st.sidebar:
                        st.write("Summarization triggered. New prompt:")
                        st.json(new_memory, expanded=False)

                # Build the output
                res_data = {
                    'messages': new_memory,
                }
                return res_status, res_message, res_data

        # If we reach here, it means that even 25% of the reply tokens still exceed the max possible tokens.
        res_status = 2
        res_message = "Summarization triggered but failed to generate a summary that fits the model's token limit."
        return res_status, res_message, res_data

    # No need to summarize, just return the original prompt
    res_data = {
        'messages': messages,
    }
    return res_status, res_message, res_data


def get_chat_message(
    i: int,
    message: dict[str, str],
    loading: bool = False,
    loading_fp: Path = FILE_ROOT / "assets" / "loading.gif",
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
        st.write("")

    with contents_container:
        st.markdown(contents)
        if len(sources) > 0:
            if streaming:
                pass
            else:
                sources = json.loads(sources)
                try:
                    urls = []
                    for source in sources["sources"]:
                        if "url" not in source:
                            raise
                        if source["url"].strip() == "":
                            raise
                        urls.append(source["url"])
                    if len(urls) > 0:
                        markdown_text = ""
                        for j, url in enumerate(urls):
                            markdown_text += f"[[{j+1}]]({url}) "
                        st.markdown(markdown_text)
                except:
                    with st.expander("Sources"):
                        st.json(sources["sources"], expanded=True)
        if i >= 0:
            copy_to_clipboard(f"copy_{i}", contents)
        if loading:
            st.markdown(f"<img src='data:image/gif;base64,{get_local_img(loading_fp)}' width=30 height=10>", unsafe_allow_html=True)



def search_documents(kwargs) -> list[str]:
    """Helper function to search documents in the provided database"""
    query = kwargs.get("query", None)
    if query is None:
        return []
    if vector_db is None:
        return []
    else:
        docs = vector_db.similarity_search(query, VECTOR_N_RESULTS)

    if DEBUG:
        with st.sidebar:
            st.write("Query")
            st.text(query)
            st.write("Reference materials")
            st.json(docs, expanded=False)

    # Output reference materials (with sources, if appropriate)
    outputs = []
    for doc in docs:
        processed_contents = doc.page_content
        if "url" in doc.metadata:
            processed_contents += f"\n\nSource: {doc.metadata['url'].rstrip('/')}"
        elif "source" in doc.metadata:
            processed_contents += f"\n\nSource: {doc.metadata['source'].rstrip('/')}"
        if "page" in doc.metadata:
            processed_contents += f" (page {doc.metadata['page']})"
        outputs.append(processed_contents)
    return outputs


FUNCTIONS = {
    "available_funs": {
        "search_documents": search_documents,
    }, 
    "api_in": [
        {
            "name": "search_documents",
            "description": "Search for relevant documents from a database to help answer questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string, inferred from the user message, to search the documents for. Sample query styles: 'maatalousvähennys verotuksessa' or '工匠师进场要留意的安全事项'"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}


async def main(human_prompt: str) -> tuple[int, str]:
    res_status = 0
    res_message = "Success"
    try:
        # Strip the prompt of any potentially harmful html/js injections
        human_prompt = human_prompt.replace("<", "&lt;").replace(">", "&gt;").strip()
        
        # Update chat log
        message = {"role": "user", "content": human_prompt}
        st.session_state.MESSAGES.append(message)

        # Clear the input box after human_prompt is used
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            message = st.session_state.MESSAGES[-1]
            get_chat_message(-1, message)

            reply_box = st.empty()
            with reply_box:
                get_chat_message(-1, {
                    "role": "assistant",
                    "content": ""
                }, loading=True)

        # Step 1: See if chat memory needs to be shortened to fit context
        summary_res, summary_message, summary_data = await generate_prompt_from_memory_async(
            messages=st.session_state.MESSAGES,
            model_name="gpt-35-turbo-16k",    # Always use gpt-3.5-turbo-16k for summarization for cost and speed reasons
            max_tokens=NLP_MODEL_MAX_TOKENS,
            functions_tokens=NLP_MODEL_FUNCTIONS_TOKENS,
            reply_max_tokens=NLP_MODEL_REPLY_MAX_TOKENS
        )
        if summary_res != 0:
            return summary_res, summary_message
        
        st.session_state.MESSAGES = summary_data["messages"]

        # Copy a short-term working memory messages list to use in this one back-and-forth
        messages = st.session_state.MESSAGES.copy()

        # Step 2: Let the model generate a reply or call any functions to help it until it's ready
        while True:
            # Before calling the actual API, for each round in step 2 we still need to track if the memory needs to be shortened
            summary_res, summary_message, summary_data = await generate_prompt_from_memory_async(
                messages=messages,
                model_name="gpt-35-turbo-16k",    # Always use gpt-3.5-turbo-16k for summarization for cost and speed reasons
                max_tokens=NLP_MODEL_MAX_TOKENS,
                functions_tokens=NLP_MODEL_FUNCTIONS_TOKENS,
                reply_max_tokens=NLP_MODEL_REPLY_MAX_TOKENS
            )

            if summary_res != 0:
                return summary_res, summary_message
            
            messages = summary_data["messages"]

            reply_status, reply_message, reply_data = await get_model_reply_async(
                messages=messages,
                model_name=NLP_MODEL_NAME,
                max_tokens=NLP_MODEL_REPLY_MAX_TOKENS,
                temperature=0,
                custom_init_prompt=INITIAL_PROMPT,
                function_call="auto",
                functions=FUNCTIONS,
                streaming=True,
                container=reply_box,
            )
            if reply_status != 0:
                return reply_status, reply_message
            if "DEBUG" in st.session_state and st.session_state.DEBUG:
                with st.sidebar:
                    st.write("reply_data")
                    st.json(reply_data, expanded=False)

            # Check if the first item in the output messages is an actual assistant message, if so then break the loop
            if reply_data["messages"][0]["role"] == "assistant":
                break

            # If role is function, and the message begins with "IMAGE_URL: " string, we convert it to an assistant reply.
            if reply_data["messages"][0]["role"] == "function" and reply_data["messages"][0]["content"].startswith("IMAGE_URL: "):
                # There might be multiple messages in the messages list that are images, go through and collect all into a single message, separated by newlines
                image_markdowns = []
                for message in reply_data["messages"]:
                    if message["content"].startswith("IMAGE_URL: "):
                        image_markdowns.append(message["content"].split("IMAGE_URL: ", 1)[1])
                reply_data["messages"] = [{"role": "assistant", "content": "\n\n".join(image_markdowns)}]
                break
                
            # Otherwise, append all output messages to the short-term working memory and continue (but to save some context, we can erase any previously returned function call messages)
            messages = [message for message in messages if message["role"] != "function"]
            messages += reply_data["messages"]

    except:
        return 2, traceback.format_exc()

    # Update message history
    st.session_state.MESSAGES += reply_data["messages"]

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
    # st.write("")
    # st.info(
    #     f"Note: This app uses OpenAI's GPT-4 under the hood. The service may sometimes be busy, so please wait patiently if the reply doesn't begin immediately.",
    #     icon="ℹ️")

if DEBUG:
    with st.sidebar:
        st.subheader("Debug Area")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()

# Initialize/maintain a chat log so we can keep tabs on previous Q&As

if "MESSAGES" not in st.session_state:
    st.session_state.MESSAGES = []

# Render chat history so far
with chat_box:
    for i, message in enumerate(st.session_state.MESSAGES):
        get_chat_message(i, message)

# Define an input box for human prompts
with prompt_box:
    with st.form(key="Text input", clear_on_submit=True):
        human_prompt = st.text_area(USER_PROMPT, value="", placeholder=USER_PROMPT, label_visibility="collapsed", key=f"text_input_{len(st.session_state.MESSAGES)}", height=100)
        submitted = st.form_submit_button(label="Send")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if submitted and len(human_prompt) > 0:
    status, message = asyncio.run(main(human_prompt))
    if status == 0 and not DEBUG:
        st.rerun()
    else:
        if status != 0:
            st.error(message)
        with prompt_box:
            if st.button("Show text input field"):
                st.rerun()
