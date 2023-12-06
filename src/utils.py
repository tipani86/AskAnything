import json
import base64
import openai
import tiktoken
import traceback
import streamlit as st
from typing import Any
from pathlib import Path
from app_config import *
from app import get_chat_message
import streamlit.components.v1 as components

FILE_ROOT = Path(__file__).parent.resolve()

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages.
    From: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k" or model=="gpt-35-turbo" or model=="gpt-35-turbo-16k":
        # print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4" or model == "gpt-4-0613":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


@st.cache_data(show_spinner=False)
def get_local_img(file_path: Path) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")


def copy_to_clipboard(id: str, text: str):
    clipboard_js = """<style>
body {{
    margin: 0px;
}}
</style>
<button id="{}" title="Copy to Clipboard">📋</button>
<script>
document.getElementById("{}").addEventListener("click", event => {{
    navigator.clipboard.writeText(`{}`);
}});
</script>"""
    components.html(clipboard_js.format(id, id, text.replace("`", "\\`")), height=26)


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
