import time
import json
import os
from openai import OpenAI

# ------------------- Main Functions -------------------

def modelscope_Think(messages, temperature=1, model="deepseek-ai/DeepSeek-R1-0528", max_retries=3, max_token=65535):
    """
    Calls a model that supports a thinking process (e.g., deepseek-reasoner).
    The function handles configuration reading and client initialization internally.
    It uses the streaming API to internally aggregate the complete thinking process and the final answer, and collects token information.
    """
    # --- 1. Read, validate configuration file and initialize the client (inlined logic) ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "LLM_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found. Please ensure the {config_path} file exists.")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "Modelscope" not in config:
            raise KeyError("The 'Modelscope' configuration item is missing in the LLM_config.json file.")

        modelscope_config = config["Modelscope"]
        url = modelscope_config.get("url")
        key = modelscope_config.get("key")

        if not url:
            raise ValueError("In the 'Modelscope' configuration, the 'url' field is missing or empty.")
        if not key:
            raise ValueError("In the 'Modelscope' configuration, the 'key' field is missing or empty.")
        
        client = OpenAI(api_key=key, base_url=url)

    except Exception as e:
        error_message = f"LLM configuration error: {str(e)}"
        print(error_message)
        return 0, 0, "", error_message

    # --- 2. Core logic for the API call ---
    attempt = 0
    token_data = {"prompt_tokens": 0, "completion_tokens": 0}

    while attempt < max_retries:
        try:
            content = ""
            reasoning_content = ""
            
            stream_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=max_token
            )

            for chunk in stream_response:
                if not chunk.choices:
                    if chunk.usage:
                        token_data["prompt_tokens"] = chunk.usage.prompt_tokens
                        token_data["completion_tokens"] = chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta
                
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif delta.content:
                    content += delta.content

            break  

        except Exception as e:
            print(f"['Think' model - Attempt {attempt + 1}] LLM call exception: {str(e)}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(1*attempt)
            else:
                return token_data["prompt_tokens"], token_data["completion_tokens"], "", "The LLM call still failed after multiple retries."

    return token_data["prompt_tokens"], token_data["completion_tokens"], reasoning_content, content


def modelscope_chat(messages, temperature=1, model="Qwen/Qwen3-235B-A22B-Instruct-2507", max_retries=3, max_token=8192):
    """
    Calls a standard chat model (e.g., deepseek-chat).
    The function handles configuration reading and client initialization internally.
    It uses the streaming API to internally aggregate the complete answer and collect token information.
    """
    # --- 1. Read, validate configuration file and initialize the client (inlined logic) ---
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "LLM_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found. Please ensure the {config_path} file exists.")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        if "Modelscope" not in config:
            raise KeyError("The 'Modelscope' configuration item is missing in the LLM_config.json file.")

        modelscope_config = config["Modelscope"]
        url = modelscope_config.get("url")
        key = modelscope_config.get("key")

        if not url:
            raise ValueError("In the 'Modelscope' configuration, the 'url' field is missing or empty.")
        if not key:
            raise ValueError("In the 'Modelscope' configuration, the 'key' field is missing or empty.")
        
        client = OpenAI(api_key=key, base_url=url)

    except Exception as e:
        error_message = f"LLM configuration error: {str(e)}"
        print(error_message)
        return 0, 0, "", error_message

    # --- 2. Core logic for the API call ---
    attempt = 0
    token_data = {"prompt_tokens": 0, "completion_tokens": 0}

    while attempt < max_retries:
        try:
            content = ""
            
            stream_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=max_token
            )

            for chunk in stream_response:
                if not chunk.choices:
                    if chunk.usage:
                        token_data["prompt_tokens"] = chunk.usage.prompt_tokens
                        token_data["completion_tokens"] = chunk.usage.completion_tokens
                    continue
                
                delta = chunk.choices[0].delta

                if delta.content:
                    content += delta.content

            break

        except Exception as e:
            print(f"['Chat' model - Attempt {attempt + 1}] LLM call exception: {str(e)}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(1*attempt)
            else:
                return token_data["prompt_tokens"], token_data["completion_tokens"], "", "The LLM call still failed after multiple retries."

    return token_data["prompt_tokens"], token_data["completion_tokens"], "", content