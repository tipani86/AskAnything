import asyncio
import traceback
from app_config import *


async def call_post_api_async(
    httpclient,
    url: str,
    headers: dict = None,
    data: dict = None,
) -> dict:
    res = {'status': 0, 'message': 'success', 'data': None}

    # Make an async post request to the API with timeout, retry, backoff etc.
    for i in range(N_RETRIES):
        try:
            if DEBUG:
                print(f"Attempt {i+1}: Calling API {url} with data {data}")
            async with httpclient.post(url, headers=headers, json=data, timeout=TIMEOUT) as response:
                if response.status == 200:
                    res['data'] = await response.json()
                    return res
                else:
                    if i == N_RETRIES - 1:
                        res['status'] = 2
                        res['message'] = f"API returned status code {response.status} and message {await response.text()} after {N_RETRIES} retries."
                        return res
                    else:
                        await asyncio.sleep(COOLDOWN + BACKOFF ** i)
        except:
            if i == N_RETRIES - 1:
                res['status'] = 2
                res['message'] = f"API call failed after {N_RETRIES} tries: {traceback.format_exc()}"
                return res
            else:
                await asyncio.sleep(COOLDOWN + BACKOFF ** i)

    res['status'] = 2
    res['message'] = f"Failed to call API after {N_RETRIES} retries."
    return res


async def get_chatbot_reply_data_async(
    httpclient,
    messages: list,
    api_key: str,
) -> dict:
    res = {'status': 0, 'message': "success", 'data': None}

    # Call the OpenAI API
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': "application/json",
        'Authorization': f"Bearer {api_key}",
    }
    data = {
        'model': NLP_MODEL_NAME,
        'messages': messages,
        'max_tokens': NLP_MODEL_REPLY_MAX_TOKENS,
        'stop': NLP_MODEL_STOP_WORDS,
    }
    api_res = await call_post_api_async(httpclient, url, headers, data)
    if api_res['status'] != 0:
        res['status'] = api_res['status']
        res['message'] = api_res['message']
        return res

    reply_text = api_res['data']['choices'][0]['message']['content'].strip()

    res['data'] = reply_text
    return res
