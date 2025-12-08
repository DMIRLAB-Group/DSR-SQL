import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from transformers import AutoTokenizer, logging
import os
import glob

logging.set_verbosity_error()

def truncate_text_by_tokens(text, max_tokens=4096):
    """
    Truncates the text so that its token count does not exceed max_tokens, and returns the truncated string.
    Automatically uses the tokenizer from the current script's directory.

    Args:
        text (str): The original string.
        max_tokens (int): The maximum number of tokens to truncate to.

    Returns:
        str: The truncated string.
    """
    # Get the directory of the current Python script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chat_tokenizer_dir = current_dir  # The tokenizer path is the current directory

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=False)

    # Use truncation=True to truncate the input
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    # Decode back to the truncated string
    truncated_text = tokenizer.decode(inputs["input_ids"], skip_special_tokens=True)
    return truncated_text


def get_token_count(text: str) -> int:
    """
    Calculates the number of tokens in a text.
    Automatically uses the tokenizer from the current script's directory.

    Args:
        text (str): The original string for which to calculate the token count.

    Returns:
        int: The number of tokens corresponding to the text.
    """
    # Get the directory of the current Python script
    # Note: In an interactive environment (like Jupyter), __file__ may not be defined.
    # In such cases, you might need to specify the path manually or use os.getcwd().
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If running in an interactive environment where __file__ is not defined, use the current working directory
        current_dir = os.getcwd()
        
    tokenizer_dir = current_dir  # The tokenizer path is the current directory

    # Load the tokenizer
    # Ensure tokenizer files (tokenizer.json, etc.) exist in this directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=False)

    # Use tokenizer.encode() to encode the text, which returns a list of token IDs
    token_ids = tokenizer.encode(text)

    # The length of the list is the number of tokens
    return len(token_ids)


SL='''
We are given a problem and a previous SQL query that was executed. We need to analyze the SQL based on the given steps and the reference columns.
'''


def analyze_all_markdown_files(directory):
    """
    Iterates through all .md files in the specified directory, reads their content,
    and outputs the token count for each file.

    Args:
        directory (str): The path to the directory containing the .md files.
    """
    # Use glob to find all .md files
    md_files = glob.glob(os.path.join(directory, "*.md"))
    
    if not md_files:
        print(f"No .md files found in the directory: {directory}")
        return

    print(f"Found {len(md_files)} Markdown files, calculating token counts...\n")
    
    for file_path in sorted(md_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            token_count = get_token_count(content)
            print(f"{os.path.basename(file_path)} | Tokens: {token_count}")
        except Exception as e:
            print(f"{os.path.basename(file_path)} | Failed to read: {e}")

# Main program entry point
if __name__ == "__main__":

    # Analyze all .md files in the entire directory
    doc_dir = "/spider2-snow/resource/documents"
    analyze_all_markdown_files(doc_dir)