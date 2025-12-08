## For extracting tables and columns in Snowflake SQL, if you have a better approach, please let me know!
## This tool-related component can use a very small language model to speed up processing.
import sys
import os,time
import pickle 
import traceback
from typing import List, Tuple, Dict

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)

from LLM.LLM_OUT import LLM_output
from utils.extract_json import *
from utils.Database_Interface import snow_DB_dir,sqlite_DB_dir


# Snowflake and Bigquery share the same organizational structure.
def get_prompt(SQL,db_type="snow"):
    if db_type=="snow" or db_type=="bigquery":
        prompt=f'''
### For the current series of Snowflake SQL , please help me extract the entity tables and columns used in it, excluding virtual tables, CTEs (Common Table Expressions), aliases, etc. Upon completion, I will give you a $100 tip!  
* Note: The database name.schema name.table name must be complete (none of the three parts can be missing).  
* Note: You should analyze and identify the column names that actually exist in the database.  
* Note: Distinguish between column names and JSON key names, and do not mistakenly treat key names as column names. 
Please retain clean table names and column names, which means you should remove double quotes, escape characters, and the like. 
All SQL information can be combined and placed in a single JSON block. 
## SQL:
{SQL}

## Output Format:
```json
{{
"Databasename.Schemaname.Tablename": ["Column1", "Column2", ……]
}}
```

Take a deep breath and think step by step!
'''
    if db_type=="sqlite":
        prompt=f'''
### For the current series of SQLite SQL, please help me extract the entity tables and columns used in it, excluding virtual tables, CTEs (Common Table Expressions), aliases, etc. Upon completion, I will give you a $100 tip!
* Note: You should analyze and identify the column names that actually exist in the database.
* Please retain clean table names and column names, which means you should remove double quotes, escape characters, and the like.
* Some table names may contain special characters such as spaces (but they are not dirty symbols left by escaping), please keep them as they are! 
All SQL information can be combined and placed in a single JSON block.
## SQL:
{SQL}

## Output Format:
```json
{{
"Tablename": ["Column1", "Column2", ……]
}}
```
Please provide the answer in two steps of thinking:
- All tables used by the SQL
- Columns used by each table
'''

    return prompt

#TODO The current logic for determining table correctness needs to be revised. Currently, it does not evaluate table correctness at all and relies solely on consistency.
def Get_SL_func_snow(SQL, db_name, model="deepseek-chat",
                allow_partial=False, check_columns: bool = False, db_type="snow"):  # ✅ New switch added

    def run_llm():
        """
        Internal function to prompt the LLM to extract tables and columns from the SQL.
        """
        SQL_mess = [{"role": "user", "content": get_prompt(SQL=SQL, db_type=db_type)}]
        _, _, _, LLM_return = LLM_output(
            messages=SQL_mess,
            model=model,
            temperature=0,
            max_token=4096
        )
        print(LLM_return)
        return LLM_return

    max_retries = 10
    attempt = 0
    last_table_col = None
    matched_cols = set()
    unmatched_cols = set()

    while attempt < max_retries:
        attempt += 1
        try:
            LLM_return = run_llm()
            # Parse the JSON output from LLM into a dictionary {table: [col1, col2]}
            table_col = extract_and_parse_json(LLM_return)
            
            if table_col == {}: 
                return {}  # The current table extraction is empty/useless
            
            last_table_col = table_col

            if not check_columns:
                # ✅ Only identify tables, skip column correction/validation, return immediately.
                # Returns tables with empty column lists.
                return {table: [] for table in table_col.keys()}

            # ----------- Normal Column Correction Logic -----------
            # Note: This logic assumes 'all_cols_lower' is defined (DB schema), 
            # intended to verify if the LLM-predicted columns actually exist.
            expanded_cols = {f"{t}.{c}" for t, cols in table_col.items() for c in cols}
            current_matched = {col for col in expanded_cols if col.lower() in all_cols_lower}
            current_unmatched = expanded_cols - current_matched

            matched_cols.update(current_matched)
            unmatched_cols = current_unmatched

            if not unmatched_cols:
                print(f"[Get_SL] Attempt {attempt}: All columns matched successfully.")
                break
            print(f"[Get_SL] Attempt {attempt}: Some columns not matched {unmatched_cols}, retrying...")

        except Exception as e:
            traceback.print_exc()
            print(f"[Get_SL] Attempt {attempt}: Error during processing - {e}")
            continue

    # --- After loop finishes, process final results ---
    final_table_col = {}
    source_table_col = last_table_col if last_table_col else {}

    if check_columns:
        # Load the ground truth database schema (pickle file)
        db_dir = os.path.join(snow_DB_dir, db_name)
        file_path = os.path.join(db_dir, f"{db_name}_all_col.pkl")
        with open(file_path, 'rb') as file:
            all_cols = pickle.load(file)
        all_cols_lower = {c.lower() for c in all_cols}
    
        # Filter logic: Keep only columns that were successfully matched against the DB schema
        if allow_partial or not unmatched_cols:
            if last_table_col:
                for table, cols in source_table_col.items():
                    valid_cols = [col for col in cols if f"{table}.{col}" in matched_cols]
                    if valid_cols:
                        final_table_col[table] = valid_cols
        else:
            # If partial matching is not allowed and there are errors, handle accordingly
            print(f"[Get_SL] After {max_retries} attempts, the following columns couldn't be matched: {unmatched_cols}")
            print("[Get_SL] Discarding unmatched columns as allow_partial is False.")
            if last_table_col:
                for table, cols in source_table_col.items():
                    valid_cols = [col for col in cols if f"{table}.{col}" in matched_cols]
                    if valid_cols:
                        final_table_col[table] = valid_cols

        # Fallback: If no columns matched, return the empty structure of the last identified tables
        if not final_table_col and last_table_col:
            print(f"[Get_SL] After all attempts, no valid columns were matched. Returning last parsed structure (might be empty).")
            final_table_col = {table: [] for table in last_table_col.keys()}
    else:
        # ✅ Only table existence verification (implied), no column processing
        if last_table_col:
            final_table_col = {table: [] for table in last_table_col.keys()}

    return final_table_col

def Get_SL_func_sqlite(SQL, db_name, model="deepseek-chat",
                allow_partial=False, check_columns: bool = True, db_type="sqlite"):
    """
    Call LLM to extract tables and columns used in SQL and validate them against the database Schema.

    Args:
        SQL (str): The input SQL query.
        db_name (str): Database name, used to locate the schema JSON file and the top-level key within it.
        model (str): The LLM model to use.
        allow_partial (bool): (Not used in this implementation, but kept for signature compatibility).
        check_columns (bool): Whether to validate column names.

    Returns:
        dict: Validated dictionary of tables and columns. Returns an empty dict if failure.
    """
    # 1. Import and preprocess Database Schema
    db_json_path = f"{sqlite_DB_dir}/{db_name}/{db_name}_M-Schema.json" 
    try:
        with open(db_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{db_json_path}'")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file '{db_json_path}'")
        return {}
    
    schema_tables = data.get(db_name, {})
    
    if not schema_tables:
        print(f"Error: Key '{db_name}' not found in JSON file or value is empty.")
        return {}

    # Create lowercase-to-original mapping for case-insensitive lookup
    schema_tables_lower_map = {t.lower(): t for t in schema_tables.keys()}
    
    schema_cols_lower_map = {}
    if check_columns:
        for table_name, cols_data in schema_tables.items():
            # Store column mappings for each table
            schema_cols_lower_map[table_name] = {col_info[0].lower(): col_info[0] for col_info in cols_data}

    def run_llm():
        print(get_prompt(SQL=SQL, db_type=db_type))
        SQL_mess = [{"role": "user", "content": get_prompt(SQL=SQL, db_type=db_type)}]
        _, _, _, LLM_return_str = LLM_output(
            messages=SQL_mess,
            model=model,
            temperature=0,
            max_token=2048
        )
        print("LLM_return_str: ", LLM_return_str)
        return LLM_return_str

    # 2. Retry Loop
    MAX_RETRIES = 10
    for attempt in range(MAX_RETRIES):
        print(f"\n--- Attempt {attempt + 1}/{MAX_RETRIES} ---")
        try:
            # 3. Call LLM and parse
            LLM_return = run_llm()
            table_col_from_llm = extract_and_parse_json(LLM_return)
            
            if not table_col_from_llm or not isinstance(table_col_from_llm, dict):
                print("LLM return is empty or format is incorrect, retrying...")
                time.sleep(1)
                continue

            # 4. Validate tables and columns
            validated_result = {}
            for llm_table, llm_cols in table_col_from_llm.items():
                llm_table_lower = llm_table.lower()

                # 4.1 Validate table name
                if llm_table_lower not in schema_tables_lower_map:
                    print(f"Warning: Table '{llm_table}' returned by LLM not found in DB '{db_name}', discarding.")
                    continue
                
                original_table_name = schema_tables_lower_map[llm_table_lower]
                
                # 4.2 If column check is not required, add table name directly
                if not check_columns:
                    validated_result[original_table_name] = []
                    continue

                # 4.3 Validate column names (if check_columns is True)
                validated_cols = []
                if not isinstance(llm_cols, list):
                    print(f"Warning: Columns for table '{llm_table}' is not a list, ignoring.")
                    llm_cols = []

                # Retrieve the column map for the matched original table name
                current_table_cols_map = schema_cols_lower_map[original_table_name]
                
                for llm_col in llm_cols:
                    llm_col_lower = llm_col.lower()
                    if llm_col_lower in current_table_cols_map:
                        original_col_name = current_table_cols_map[llm_col_lower]
                        validated_cols.append(original_col_name)
                    else:
                        print(f"Warning: Column '{llm_col}' returned by LLM not found in table '{original_table_name}', discarding.")
                
                validated_result[original_table_name] = validated_cols
            
            # 5. Check validation results
            if validated_result:
                print("Successfully retrieved and validated tables and columns.")
                return validated_result
            else:
                print("All tables or columns returned by LLM are invalid, retrying...")
                time.sleep(1)

        except Exception as e:
            print(f"Exception occurred during attempt: {e}, retrying...")
            time.sleep(1)

    print(f"All {MAX_RETRIES} attempts failed.")
    return {}