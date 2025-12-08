import re
import json
import json_repair  # JSON repair utility

def extract_and_parse_json(text: str) -> dict:#DeepSeek V3.1
    """
    Robustly extracts, repairs, and parses a JSON object from a text string.

    This function is designed to handle all of the following challenging situations:
    - Standard JSON enclosed in ```json ... ``` code blocks.
    - Bare JSON objects without code block fences.
    - Extraneous whitespace characters before or after the JSON object.
    - Non-JSON trailing characters after the JSON object.
    - Syntax errors within the JSON (e.g., missing commas, extra commas, use of single quotes).
    - **Truncated LLM output, resulting in an incomplete JSON structure (e.g., missing a final '}' or ']').**

    Processing Flow:
    1.  **First, attempt to match a complete ```json ... ``` code block.**
    2.  **If that fails, attempt to match a potentially truncated ```json code block (i.e., only looking for the start).**
    3.  **If that also fails, find the first '{' or '[' in the text and extract all subsequent content from that point.**
    4.  The extracted (and possibly incomplete) string is then passed to the powerful `json_repair` library for fixing and parsing.

    Args:
        text (str): The raw text string containing the JSON.

    Returns:
        dict: The parsed Python dictionary.

    Raises:
        ValueError: If no JSON-like structure can be found in the text, or if the repair and parsing process ultimately fails.
    """
    
    json_str_candidate = None

    # Strategy 1: Look for a complete ```json ... ``` code block.
    # This is the ideal and most explicit case.
    match = re.search(r'```json\s*({[\s\S]*?})\s*```', text, re.DOTALL)
    if match:
        json_str_candidate = match.group(1)
    
    # Strategy 2: If a complete block isn't found, look for one that starts with ```json but might be truncated.
    if not json_str_candidate:
        match = re.search(r'```json\s*({[\s\S]*)', text, re.DOTALL)
        if match:
            json_str_candidate = match.group(1)

    # Strategy 3: If neither of the above code blocks is found, look for the first '{' or '[', treating it as the start of a bare JSON object.
    if not json_str_candidate:
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        # Determine which one appears first
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            json_str_candidate = text[start_brace:]
        elif start_bracket != -1:
            json_str_candidate = text[start_bracket:]

    # If all strategies have failed, it means no JSON structure was found.
    if not json_str_candidate:
        raise ValueError("No JSON-like structure (including incomplete ones) was found in the text.")

    # Clean the string and use json_repair, as it can handle all cases (complete, malformed, and truncated).
    cleaned_str = json_str_candidate.strip()
    
    try:
        # json_repair is the best tool for handling this kind of uncertain input.
        return json_repair.loads(cleaned_str)
    except Exception as e:
        # If even json_repair fails, the input is truly problematic.
        raise ValueError(f"JSON repair and parsing failed. The string attempted was: '{cleaned_str}'. Error: {e}")

def extract_sql(text: str) -> str:
    """
    Robustly extracts an SQL code block from a text string.

    This function can handle the following situations:
    - Standard SQL enclosed in ```sql ... ``` code blocks.
    - Truncated LLM output where the SQL code block is missing the closing ```.

    Processing Flow:
    1.  **First, attempt to match a complete ` ```sql ... ``` ` code block.** This is the ideal case.
    2.  **If that fails, attempt to match a potentially truncated ` ```sql ` code block** (i.e., it only looks for the start and captures everything that follows).
    3.  **If all strategies fail, a `ValueError` is raised.**

    Args:
        text (str): The raw text string containing the SQL code block.

    Returns:
        str: The cleaned and extracted SQL query string.

    Raises:
        ValueError: If no code block starting with ` ```sql ` can be found in the text.
    """
    sql_candidate = None

    # Strategy 1: Look for a complete ```sql ... ``` code block.
    # This is the ideal and most explicit case. Using \s* instead of \s+ handles cases with no space, like ```sql...```
    pattern_complete = r"```sql\s*(.*?)\s*```"
    match_complete = re.search(pattern_complete, text, re.DOTALL | re.IGNORECASE)
    
    if match_complete:
        sql_candidate = match_complete.group(1).strip()
    
    # Strategy 2: If a complete block isn't found, look for one that starts with ```sql but might be truncated.
    if not sql_candidate:
        pattern_partial = r"```sql\s*(.*)"
        match_partial = re.search(pattern_partial, text, re.DOTALL | re.IGNORECASE)
        if match_partial:
            # Even if it might be truncated, we still take its content.
            sql_candidate = match_partial.group(1).strip()

    # If all strategies have failed, it means no SQL was found.
    if not sql_candidate:
        raise ValueError("Extraction failed! No valid SQL code block was found in the text.")

    return sql_candidate


def extract_answer_content(text: str) -> str: # Frequent issue with DEEPSEEK V31
    """
    Extracts content wrapped in <answer> tags from text, supporting incomplete tags.

    Matching Priority:
        1. <answer> ... </answer>
        2. <answer> ... <
        3. <answer> ... (until the end of the text)

    Args:
        text (str): The input text.

    Returns:
        str: The extracted content (with leading/trailing whitespace removed).

    Raises:
        ValueError: If the <answer> tag is not found or the content is empty.
    """
    # 1. Attempt to match a complete <answer> ... </answer> tag.
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # 2. Attempt to match an incomplete closing tag: <answer> ... <
    match = re.search(r"<answer>(.*?)<", text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # 3. As a fallback, match all content after the opening <answer> tag.
    match = re.search(r"<answer>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # If all attempts fail, raise an error.
    raise ValueError("No content wrapped in <answer> tags was found, or the content was empty.")



if __name__ == "__main__":

    text="""
{'current_subquestion': "Identify the CustomerID from the transactions_1k table where Amount equals 635 and Date equals '2012-08-25' (adjusted for integer Amount type).", 'current_subsql': "SELECT CustomerID FROM transactions_1k WHERE Amount = 635 AND Date = '2012-08-25'", 'is_final_subquestion': False, 'solved_subquestions_list': ["Identify the CustomerID from the yearmonth table where Consumption equals 634.8 and Date equals '201208'", "Identify the CustomerID from the transactions_1k table where Amount equals 634.8 and Date equals '2012-08-25'", "Identify the CustomerID from the yearmonth table where Consumption is approximately 634.8 and Date equals '201208' using floating-point tolerance", "Identify the CustomerID from the transactions_1k table where Amount equals 635 and Date equals '2012-08-25' (adjusted for integer Amount type)"]}
```
    """
    temp=extract_and_parse_json(text=text)
    print(temp,type(temp))

    text = """
    TEXT TO SQL：
    <Answer>
    BIRD | Spider2.0
    </answer>
    """

    result = extract_answer_content(text)
    print(result)


