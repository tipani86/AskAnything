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
