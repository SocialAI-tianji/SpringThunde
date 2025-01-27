import requests
import time
import os

from openai import OpenAI
base_url = os.getenv("OPENAI_API_BASE_URL") if os.getenv(
    "OPENAI_API_BASE_URL") is not None else "https://api.openai.com/v1"
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME") if os.getenv(
    "OPENAI_MODEL_NAME") is not None else "gpt-4o"
client = OpenAI(
    api_key=api_key,
    base_url=base_url)


def get_oai_completion(prompt):
    """
    Get completion from OpenAI API.
    
    Args:
        prompt (str): The input prompt to send to the OpenAI API
        
    Returns:
        str: The generated text response from the model
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(model=model_name,
                                              messages=messages,
                                              temperature=1,
                                              max_tokens=2048,
                                              top_p=0.95,
                                              frequency_penalty=0,
                                              presence_penalty=0,
                                              stop=None)
    res = response.choices[0].message.content

    gpt_output = res
    return gpt_output


def call_chatgpt(ins):
    """
    Call ChatGPT API with retry mechanism.
    
    Args:
        ins (str): The instruction/prompt to send to ChatGPT
        
    Returns:
        str: The response from ChatGPT. Returns empty string if all retries fail
    """
    success = False
    re_try_count = 15
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        # try:
        ans = get_oai_completion(ins)
        success = True
        # except:
        #     time.sleep(5)
        #     print('retry for sample:', ins)
    return ans
