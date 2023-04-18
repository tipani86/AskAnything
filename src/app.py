# App to load the vector database and let users to ask questions from it
import os
import base64
import aiohttp
import asyncio
import argparse
import traceback
from utils import *
import streamlit as st
from app_config import *
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--site", type=str, required=True, help="The site to load (section name in cfg file)")

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    print(f"Error parsing command line arguments: {traceback.format_exc()}")
    os._exit(e.code)


### FUNCTION DEFINITIONS ###


@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(FILE_ROOT, "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"


@st.cache_data(show_spinner=False)
def get_local_img(file_path: str) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


def get_chat_message(
    contents: str = "",
    align: str = "left"
) -> str:
    # Formats the message in an chat fashion (user right, reply left)
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    file_path = os.path.join(FILE_ROOT, "assets", "AI_icon.png")
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

        # Update both chat log and the model memory
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
            docs = vector_db.similarity_search(human_prompt)

            messages = [
                {'role': "system", 'content': INITIAL_PROMPT}
            ] + [
                {'role': "system", 'content': f"{x.page_content}\n\n({x.metadata['source']})"} for x in docs
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
                        st.write("chatbot_res")
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


# Create a wide Streamlit UI and a custom title
st.set_page_config(
    page_title=f"Kysy karhulta",
)

# Define main layout
st.title(f"Moi, olen karhu. Verokarhu.")
st.subheader("Kysy minulta mitä tahansa verotukseen liittyvästä!")
st.subheader("")
chat_box = st.container()
st.write("")
prompt_box = st.empty()
footer = st.container()

# Load CSS code
st.markdown(get_css(), unsafe_allow_html=True)

# Load the vector database
persist_directory = os.path.join(FILE_ROOT, CHROMA_DB_DIR, args.site.replace(".", "_"))

if not os.path.exists(persist_directory):
    st.error(f"Vector database not found at {persist_directory}")
    st.stop()

embeddings = OpenAIEmbeddings()
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


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
    human_prompt = st.text_input("Kysymyksesi:", value="", key=f"text_input_{len(st.session_state.LOG)}")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(human_prompt) > 0:
    run_res = asyncio.run(main(human_prompt))
    if run_res['status'] == 0 and not DEBUG:
        st.experimental_rerun()

    else:
        if run_res['status'] != 0:
            st.error(run_res['message'])
        with prompt_box:
            if st.button("Show text input field"):
                st.experimental_rerun()
