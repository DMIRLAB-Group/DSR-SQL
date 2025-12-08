from LLM.DeepSeek_LLM import *
from LLM.Modelscope_LLM import *

def LLM_output(messages, temperature=1, model="deepseek-reasoner", max_retries=10,max_token=65535,**kwargs):
    if model in ["deepseek-reasoner","deepseek-chat"]:
        return DS_output(messages=messages,temperature=temperature,model=model,max_retries=max_retries,max_token=max_token if model == "deepseek-reasoner" else 8192)
    if model in ["Qwen/Qwen3-Coder-480B-A35B-Instruct","deepseek-ai/DeepSeek-R1-0528","Qwen/Qwen3-235B-A22B-Thinking-2507"]:
        return modelscope_Think(messages=messages,temperature=temperature,model=model,max_retries=max_retries,max_token=max_token)
    if model in ["Qwen/Qwen3-Next-80B-A3B-Instruct","Qwen/Qwen3-235B-A22B-Instruct-2507","Qwen/Qwen3-30B-A3B-Instruct-2507"]:
        return modelscope_chat(messages=messages,temperature=temperature,model=model,max_retries=max_retries,max_token=8192)
    else:
        raise ValueError(f"Error: You have not configured the corresponding LLM: '{model}'. Please check if the model name is spelled correctly.")
    

if __name__ == "__main__":

    messages = [{"role": "user", "content": "I have no wings, but I can fly you to distant lands. I have no mouth, but I can tell stories from ages past or yet to come. I am usually silent, unless someone breaks my silence. Once my silence is broken, I can never be made whole again. What am I?"}]

    input_token_count, output_token_count, reasoning, answer = LLM_output(messages, model="Qwen/Qwen3-Next-80B-A3B-Instruct")
    print("Input tokens:", input_token_count)
    print("Output tokens:", output_token_count)
    print("Reasoning process:\n<think>", reasoning,"\n</think>")
    print("Final answer:", answer)

"""
python -m LLM.LLM_OUT
"""