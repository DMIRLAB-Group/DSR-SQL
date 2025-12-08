import time
import json
import os
from openai import OpenAI

def DS_output(messages, temperature=1, model="deepseek-reasoner", max_retries=3, max_token=8192):
    """
    Initializes the client by reading a configuration file, calls the language model API, supports retries,
    and returns token usage and model output.

    Parameters:
    messages (list): The list of conversation messages to send to the model.
    temperature (float): Controls the randomness of the generated text.
    model (str): The name of the model to use.
    max_retries (int): The maximum number of retries after a failure.
    max_token (int): Specifies the maximum number of tokens for the model to generate.

    Returns:
    tuple: A tuple containing four values (input_token_count, output_token_count, reasoning_content, content).
           If there is a configuration error or the API call fails completely, it will return a tuple with an error message.
    """
    
    # --- 1. Read and validate the configuration file ---
    try:
        # Get the directory where the current script file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "LLM_config.json")

        # Check if the configuration file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found. Please ensure the {config_path} file exists.")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Check if the 'DeepSeek-AI' configuration exists
        if "DeepSeek-AI" not in config:
            raise KeyError("The 'DeepSeek-AI' configuration item is missing in the LLM_config.json file.")

        deepseek_config = config["DeepSeek-AI"]
        
        # Check if url and key exist and are not empty
        url = deepseek_config.get("url")
        key = deepseek_config.get("key")

        if not url: # Covers cases where the key is missing or the value is an empty string ""
            raise ValueError("In the 'DeepSeek-AI' configuration, the 'url' field is missing or empty.")
        
        if not key:
            raise ValueError("In the 'DeepSeek-AI' configuration, the 'key' field is missing or empty.")

        # --- 2. Initialize the API client ---
        client = OpenAI(api_key=key, base_url=url)

    except Exception as e:
        # Capture all exceptions during the configuration phase and format the return
        error_message = f"LLM configuration error: {str(e)}"
        print(error_message)
        return 0, 0, "", error_message

    # --- 3. Core logic for API calls (with retry mechanism) ---
    attempt = 0
    success_flag = False
    content = "LLM call error"
    reasoning_content = ""
    token_data = {
        "model": model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_token,
                stream=False
            )

            token_data = {
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            if model == "deepseek-reasoner":
                content = response.choices[0].message.content
                reasoning_content = response.choices[0].message.reasoning_content
            else:
                content = response.choices[0].message.content
                reasoning_content = ""
                
            success_flag = True
            break

        except Exception as e:
            print(f"[Attempt {attempt + 1}] LLM call exception: {str(e)}")
            attempt += 1
            if attempt < max_retries:
                time.sleep(1*attempt)
    
    if not success_flag:
        content = "The LLM call still failed after multiple retries."

    # --- 4. Prepare and return the results ---
    input_token_count = token_data["prompt_tokens"]
    output_token_count = token_data["completion_tokens"]
    
    return input_token_count, output_token_count, reasoning_content, content