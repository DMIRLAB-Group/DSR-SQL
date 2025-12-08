If you need to use another LLM to reproduce the workflow, you can create a corresponding function that, after calling the LLM, returns the following four variables in this exact order:

Input token count, output token count, reasoning content, and response content.

If any of the first three values are unavailable, simply use placeholders: 0, 0, and None.

Finally, please import the newly created function into `LLM_OUT.py`!

> Since we are not using GPT series or models related to openrouter, please refer to the configuration in 'DeepSeek_LLM.py' as needed.