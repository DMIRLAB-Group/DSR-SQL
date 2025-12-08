#--------------------------------  
# Here we perform table-level schema linking, i.e., using LLMs to determine whether each current table is helpful in answering the user's question, with the aim of excluding irrelevant tables.  
# Reference: https://github.com/Snowflake-Labs/ReFoRCE/blob/o3/methods/SL/gen_sl/README.md   
#--------------------------------
import sys
import os
import re
import json
import time
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

# 3. 将当前目录加入 sys.path (为了找到同级目录的 Extract_tables_col)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 4. 将项目根目录加入 sys.path (为了找到 utils.Database_Interface)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------------------------------------------------------
# 修复路径代码块 END
# ---------------------------------------------------------

# 现在路径设置好了，再进行导入
from Extract_tables_col import *
from collections import defaultdict
from datetime import datetime
import traceback
import sqlite3
from typing import List, Optional


# 将项目根目录加入 sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
)
from utils.Database_Interface import snow_DB_dir,M_Schema,generate_ddl_from_json,detect_db_type,sqlite_DB_dir,bigquery_DB_dir
from utils.app_logs.logger_config import setup_logger, log_context,JsonLogger
from utils.mytoken.deepseek_tokenizer import *

def log_llm_io(model_name: str, prompt: str, output: str, think, qid, log_file=None):
    """
    Logs LLM input and output to a file and ensures immediate disk write.
    
    Default behavior: 
    If log_file is None, it saves to: [current_script_dir]/LOG/LLM_call_SL.log
    It will automatically create the 'LOG' folder if it does not exist.
    """
    try:
        # --- Path Configuration ---
        if log_file is None:
            # Get the absolute directory where this python script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path: .../current_dir/LOG/LLM_call_SL.log
            log_file = os.path.join(current_dir, "LOG", "LLM_call_SL.log")

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Question_id": qid,
            "model": model_name,
            "prompt": prompt,
            "think": think,
            "output": output
        }

        # --- Directory Creation ---
        # Get the folder path (e.g., .../LOG) and create it if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # --- File Writing ---
        # Use 'a' (append) mode + flush/fsync to ensure data is safe even if the program crashes
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            f.flush()             # Flush the internal Python buffer
            os.fsync(f.fileno())  # Force write to the physical disk

    except Exception as e:
        print(f"[Error] Failed to write log: {e}")
        traceback.print_exc()

def simplify_table_series(all_need):
    """
    For the dictionary `all_need` containing table keys:
    1. If keys are identical after removing digits;
    2. Or if they differ only by case;
    Then keep only the first key appearing in that series and remove the rest.
    """
    # Stores the processed results
    simplified = {}
    # Tracks processed series (table names after lowercasing and digit removal)
    seen_series = {}

    # Sort keys to ensure consistent processing order (optional)
    for key in sorted(all_need.keys()):
        # Convert to lowercase and remove digits to get the "series name"
        series_name = re.sub(r'\d+', '', key.lower())

        # If this series has already been processed, skip it
        if series_name in seen_series:
            continue
        else:
            # Otherwise, mark this series as seen and keep the value of the current key
            seen_series[series_name] = True
            simplified[key] = all_need[key]

    return simplified

def extract_sql_code(text): # Extract a whole code block
    """
    Extracts the SQL code block wrapped in ```sql ... ``` from the text.
    Only matches the last occurrence.
    
    Args:
        text (str): Text containing SQL code blocks.
    Returns:
        str: The extracted SQL code (whitespace stripped). Returns None if not found or if an error occurs.
    """
    try:
        sql_pattern = r'```sql(.*?)```'
        # Find all matches, DOTALL allows . to match newlines
        sql_matches = re.findall(sql_pattern, text, re.DOTALL)
        
        if sql_matches:
            # Return the last match found
            return sql_matches[-1].strip()
        else:
            return None
    except Exception as e:
        print(f"Error occurred while extracting SQL code: {e}")
        traceback.print_exc()
        return None

def get_table_mess_snow(db_name):
    """
    Extract all non-empty table names from the specified database JSON file.
    Empty tables (values that are empty lists or dictionaries) are skipped.

    Args:
        db_name (str): The name of the database, used to construct the file path.

    Returns:
        list: A list of formatted non-empty table names (db_name.table_name).
              Returns an empty list if the file does not exist or parsing fails.
    """

    # Construct path
    db_json_path = f"{snow_DB_dir}/{db_name}/{db_name}_M-Schema.json"
    
    all_tables = []

    try:
        with open(db_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{db_json_path}'")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file '{db_json_path}'")
        return []

    # Iterate through schema
    for schema_name, schema_content in data.items():
        for key, value in schema_content.items():
            if key in ["table_Information", "table_description_summary"]:
                continue  # Skip metadata keys
            
            # Skip empty tables (could be [] or {})
            if not value:  # Empty list, empty dict, or None will be skipped
                continue

            full_table_name = f"{db_name}.{key}"
            all_tables.append(full_table_name)

    return all_tables

# --- New Function: Specifically extract table information for BigQuery ---
def get_table_mess_bigquery(db_name):
    """
    Extract full physical names of all non-empty tables from the specified BigQuery database JSON file.
    Empty tables (values that are empty lists or dictionaries) are skipped.

    Args:
        db_name (str): The logical name of the database (folder name), used to construct the file path.

    Returns:
        list: A list of formatted non-empty table names (project_id.dataset_id.table_id).
              Returns an empty list if the file does not exist or parsing fails.
    """
    # Construct BigQuery file path
    db_json_path = os.path.join(bigquery_DB_dir, db_name, f"{db_name}_M-Schema.json")
    
    all_tables = []

    try:
        with open(db_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at '{db_json_path}'")
        return []
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON file '{db_json_path}'")
        return []

    # 1. Dynamically get the top-level key (project_id)
    if not data or len(data) != 1:
        print(f"Warning: JSON file structure '{db_json_path}' unexpected (should have only one top-level key).")
        return []
    
    top_level_key = list(data.keys())[0]
    db_content = data[top_level_key]

    # 2. Iterate through datasets
    for dataset_name, dataset_content in db_content.items():
        if not isinstance(dataset_content, dict):
            continue

        # 3. Iterate through tables in the dataset
        for table_key, value in dataset_content.items():
            # Skip metadata keys
            if table_key in ["table_Information", "table_description_summary"]:
                continue
            
            # Skip empty tables (value is [], {}, None, etc.)
            if not value:
                continue

            # 4. Construct the full physical table name
            # table_key format is already "dataset_name.table_name"
            full_table_name = f"{top_level_key}.{table_key}"
            all_tables.append(full_table_name)

    return all_tables


def get_table_sqlite(db_name):
    """
    Extracts all non-empty table names from the SQLite database's JSON file.

    Args:
        db_name (str): The name of the database, used to construct the file path.

    Returns:
        list: A list containing all formatted (db_name.table_name) non-empty table names.
              Returns an empty list if the file does not exist or parsing fails.
    """
    db_json_path = f"{sqlite_DB_dir}/{db_name}/{db_name}_M-Schema.json"
    all_tables = []

    try:
        with open(db_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{db_json_path}'")
        return []
    except json.JSONDecodeError:
        print(f"❌ Error: Unable to parse JSON file '{db_json_path}'")
        return []

    # Some JSON structures are {"AdventureWorks": {...}, "foreign_keys": {...}}
    # So first find the database layer
    db_content = None
    if db_name in data:
        db_content = data[db_name]
    else:
        # Some file structures are different, automatically find the first key that is not "foreign_keys"
        for k, v in data.items():
            if k != "foreign_keys" and isinstance(v, dict):
                db_content = v
                break

    if not db_content:
        print(f"⚠️ Warning: Database structure not found in file '{db_name}'")
        return []

    # Iterate through tables under this database
    for table_name, table_content in db_content.items():
        # Skip empty tables and foreign_keys
        if table_name == "foreign_keys" or not table_content:
            continue

        full_table_name = f"{table_name}"
        all_tables.append(full_table_name)

    return all_tables

def merge_table_schemas(dict_list):
    merged_result = defaultdict(set)
    merged_foreign_keys = set()
    
    for schema_dict in dict_list:
        for table, fields in schema_dict.items():
            if table == 'foreign_keys':
                merged_foreign_keys.update(fields)
            else:
                merged_result[table].update(fields)

    # Convert sets back to lists and maintain order (optional)
    final_result = {
        table: sorted(list(fields)) for table, fields in merged_result.items()
    }
    #final_result['foreign_keys'] = sorted(list(merged_foreign_keys))
    
    return final_result

def get_tables_ddl_sqlite(db_id: str, table_list: Optional[List[str]] = None, db_type="sqlite") -> str:
    """
    Connects to the specified SQLite database, extracts the DDL for specific tables, and merges them into a single long string.

    Args:
        db_path (str): The full path to the SQLite database file.
        table_list (Optional[List[str]], optional): 
            A list containing table names.
            - If None (default), extracts all user-created tables.
            - If a list is provided, only extracts the tables specified in the list.

    Returns:
        str: A long string containing the DDLs of the selected tables, separated by ';\n\n'.
             Returns an empty string if the database does not exist, cannot be connected to, or no specified tables are found.
    """
    db_path = f"{sqlite_DB_dir}/{db_name}/{db_name}.sqlite"

    if not os.path.exists(db_path):
        print(f"Error: Database file does not exist at '{db_path}'")
        return ""

    ddl_statements = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Basic query statement
            base_query = "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            params = []

            # If table_list is provided, dynamically construct the query
            if table_list:
                # Generate placeholders (?, ?, ? ...) to safely insert parameters and prevent SQL injection
                placeholders = ', '.join(['?'] * len(table_list))
                query = f"{base_query} AND name IN ({placeholders});"
                params = table_list
            else:
                # If table_list is not provided, query all user tables
                query = base_query + ";"

            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            ddl_statements = [row[0] for row in rows if row and row[0]]

    except sqlite3.Error as e:
        print(f"Error occurred while connecting to or querying database '{db_path}': {e}")
        return ""

    if not ddl_statements:
        if table_list:
             print(f"Specified tables not found in database '{db_path}': {table_list}.")
        else:
             print(f"No user-created tables found in database '{db_path}'.")
        return ""
    
    return ';\n\n'.join(ddl_statements) + ';'



def get_prompt_SQL_old(Question, table_mess,table_name,db_type="snow"):
    db_type_temp = "Snowflake" if db_type in ("snow", "bigquery") else "Sqlite"
    prompt=f'''
You are a data analyst. Please thoroughly analyze the relationship between the user’s question and the current table structure, and write one SQL statement to maximize the potential of the current table.

* **First**, you need to analyze the purpose of each table in the current database based on its table name.
* **Second**, for the given table structure, analyze its relevance to the user’s question:

* **Direct relation**: The retrieval and calculation required to answer the question are mainly (or entirely) performed on the current table.
* **Indirect relation**: The current table may not be directly used to answer the question, but is needed to join with other tables for overall computation and answering.
* **No relation**: The content described by the current table has no relationship with the user’s question, and no potential connection is found.

## Constraints

* Assume the current table is needed, and suggest one SQL statement that fully utilizes the table’s value in answering the question (i.e., select all needed columns and perform simple SQL generation). Use `TODO` for incomplete logic.
'''
    if db_type=="snow" or db_type=="bigquery":
        prompt+=f'''
* **{db_type_temp} is case-sensitive**. All database, table, and column names must be enclosed in double quotes (`"`).
* When the provided table is part of a series, you only need to select the most relevant single table to simulate SQL generation. **Do not** `UNION ALL` all tables and **do not** use wildcards. Pick only one representative table.
'''
    prompt+=f'''
* Please ignore any information related to system tables and queries on system tables, and focus on the connections between the problem and entity tables.

## Database Table Information

{table_name}
* This may only represent a table, and there may be other tables with the same structure in the database. You can identify this by observing the **table description**.

**Input**
【Question】
{Question}

{table_mess}

**Output Format:** Strictly output the SQL in the format below.

````
<Analysis Process>
Please elaborate in detail the consideration and analysis process for each step of the problem here.
</Analysis Process>
```sql
Your SQL
````

or

```sql
SELECT FALSE AS result; -- Indicates that the current table has no relationship to the question
```
        '''
    return prompt

def get_prompt_SQL4(Question, table_mess,table_name,db_type="snow"):
    db_type_temp = "Snowflake" if db_type in ("snow", "bigquery") else "Sqlite"
    PROMPT_CE = f"""
Based on the current database schema and user question, please analyze the role of each table in detail and generate 5 to 10 diverse {db_type_temp} SQL queries.  - ranging from simple to complex types - to fully explore all possible tables and their corresponding columns that may be useful for the current question. The last query should be the possible final SQL to complete the current user question.

Each query must be unique. Avoid querying schema metadata or checking data types. Only write valid SELECT statements.

Database schema:
{table_mess}

"""
    if db_type=="snow" or db_type=="bigquery":
        PROMPT_CE += f"""

- {db_type_temp} is case-sensitive. Always enclose **all DB, table and column names in double quotes (`"`)** to avoid errors.
- When needing to perform a "UNION ALL" operation on a series of tables, it is prohibited to use wildcards such as `project_id.dataset_id.table_prefix*` for matching multiple tables in joins. **You only need to select the most relevant date table for SQL generation, and must not perform UNION ALL. That is, your task is to use a representative table for the generation exercise.**
"""
    PROMPT_CE+=f"""
Output Format:
In your answer, all SQL should be placed in a single sql code block rather than separately
```sql
-- Query1
-- Your SQL query1

-- Query2
-- Your SQL query2
...
-- Queryn
-- Your SQL queryn
```

Take a deep breath and think step by step to find the correct SQL query.
Task:
{Question}
"""
    return PROMPT_CE

def SL_workflow_old(Question_id, Question, db_id,Tool_model, model="deepseek-chat", temperature=0, max_retries=5, db_type="snow"):
    if db_type == "snow":
        table_list = get_table_mess_snow(db_id)
    elif db_type == "sqlite":
        table_list = get_table_sqlite(db_id)
    else:
        table_list = get_table_mess_bigquery(db_id)
        
    all_table_results = []

    for table in table_list:
        success = False
        retry_count = 0

        while not success and retry_count < max_retries:
            try:
                # 1. Get schema for a single table
                table_mess = M_Schema(db_id=db_id, SL=[table], db_type=db_type, Level="table")

                # 2. Construct prompt
                Prompt = get_prompt_SQL_old(Question, table_mess, str([table]), db_type=db_type)

                # 3. Call LLM
                start_time = time.time()
                input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                    messages=[{"role": "user", "content": Prompt}],
                    model=model,
                    temperature=temperature
                )
                end_time = time.time()
                elapsed_time1 = end_time - start_time
                
                print(LLM_return)
                log_llm_io(model_name=model, prompt=Prompt, output=LLM_return, think=Thinking, qid=Question_id)

                # 4. Extract SQL
                SQL = extract_sql_code(LLM_return)

                # 5. Execute SQL and parse output
                start_time = time.time()

                if db_type == "snow" or db_type == "bigquery":
                    table_x = Get_SL_func_snow(SQL=SQL, db_name=db_id, model=Tool_model,allow_partial=False, check_columns=False)
                elif db_type == "sqlite":
                    table_x = Get_SL_func_sqlite(SQL=SQL, db_name=db_id,model=Tool_model, allow_partial=False, check_columns=True)

                end_time = time.time()
                elapsed_time2 = end_time - start_time
                
                # Record the status of this run
                logger_status.log(
                    question_id=Question_id,
                    step=elapsed_time1 + elapsed_time2,
                    if_in_fix=model,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count,
                    status=None  # Mark successful processing of a table
                )

                # Collect results
                all_table_results.append(table_x)
                success = True

            except Exception as e:
                retry_count += 1
                traceback.print_exc()
                print(f"[Warning] Error processing table {table} on attempt {retry_count}: {e}")

        if not success:
            print(f"[Error] Failed to process table {table} after {max_retries} retries.")

    # Merge schemas from all tables
    try:
        merged_schema = merge_table_schemas(all_table_results)
        simplified_schema = simplify_table_series(merged_schema)
        # Maintain the same return format as SL_workflow_min
        return list(simplified_schema.keys()), simplified_schema, {}
    except Exception as e:
        print(f"[Error] Failed to merge table schemas: {e}")
        traceback.print_exc()
        # Maintain the same return format as SL_workflow_min
        return [], {}, {}

def SL_workflow_min(Question_id, Question, db_id, table_list,Tool_model,
                model="Qwen/Qwen3-Coder-480B-A35B-Instruct", 
                temperature=1, max_retries=5, max_token=50000,db_type="snow",
                use_single_table: bool = False,
                check_columns: bool = False):
    """
    Table-level linking is prone to significant redundancy, so a refinement step is performed here.
    """

    # If it is a single-table database and use_single_table is disabled, skip subsequent processes
    if len(table_list) == 1 and not use_single_table:
        print(f"[Info] Database {db_id} has only one table, skipping SQL/LLM due to use_single_table=False")
        return table_list, {table_list[0]: []}, {0: {table_list[0]: []}}   # Return empty schema and sample_history

    # Get table structure information (JSON format)
    table_mess = M_Schema(db_id=db_id, SL=table_list, db_type=db_type, Level="table")
    # Get table structure DDL information
    if db_type=="snow" or db_type=="bigquery":
        table_mess_ddl = generate_ddl_from_json(db_id,table_list,db_type=db_type)
    elif db_type=="sqlite":
        table_mess_ddl = get_tables_ddl_sqlite(db_id,table_list,db_type)

    # Calculate token count for JSON schema
    len_table_mess = get_token_count(table_mess)

    # If JSON schema exceeds token limit
    if len_table_mess > max_token:
        print(f"[Info] Schema(token count: {len_table_mess}) exceeds max_token ({max_token}). Checking DDL schema as an alternative.")

        # Calculate token count for DDL schema
        len_table_mess_ddl = get_token_count(table_mess_ddl)

        # If DDL also exceeds token limit, switch to the old workflow
        if len_table_mess_ddl > max_token:
            print(f"[Info] DDL schema(token count: {len_table_mess_ddl}) also exceeds max_token. Switching to old workflow.")
            return SL_workflow_old(
                Question_id, Question, db_id,
                model=model,
                Tool_model=Tool_model,
                temperature=temperature,
                max_retries=max_retries,
                db_type=db_type
            )
        else:
            # Use DDL schema as an alternative to the original schema
            print(f"[Info] Using DDL schema (token count: {len_table_mess_ddl}) as it is within the limit.")
            table_mess = table_mess_ddl

    # Construct Prompt input content
    Prompt = get_prompt_SQL4(Question, table_mess, str(table_list),db_type=db_type)

    # Store results for each sampling round
    all_samples = []
    # Record structure returned by each sampling
    sample_history = {}

    # Perform 3 rounds of sampling
    for sample_index in range(3):
        success = False
        retry_count = 0
        all_table = []

        # Attempt up to max_retries times per round
        while not success and retry_count < max_retries:
            try:
                start_time = time.time()

                # Call LLM to get output
                input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                    messages=[{"role": "user", "content": Prompt}],
                    model=model,
                    temperature=temperature
                )

                print(LLM_return)
                log_llm_io(model_name=model, prompt=Prompt, output=LLM_return, think=Thinking, qid=Question_id)

                end_time = time.time()
                elapsed_time1 = end_time - start_time

                # Extract SQL statement
                SQL = extract_sql_code(LLM_return)

                start_time = time.time()

                # Execute SQL to get structure information
                if db_type == "snow" or db_type=="bigquery":
                    table_x = Get_SL_func_snow(SQL=SQL, db_name=db_id,model=Tool_model, check_columns=check_columns)
                elif db_type == "sqlite":
                    table_x = Get_SL_func_sqlite(SQL=SQL, db_name=db_id, model=Tool_model,check_columns=True)


                end_time = time.time()
                elapsed_time2 = end_time - start_time

                # Record the status of this run
                logger_status.log(
                    question_id=Question_id,
                    step=elapsed_time1 + elapsed_time2,
                    if_in_fix=model,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count,
                    status=None
                )

                # If successful, record results and exit loop
                all_table.append(table_x)
                sample_history[sample_index] = table_x
                success = True

            except Exception as e:
                retry_count += 1
                traceback.print_exc()
                print(f"[Warning] Error processing all tables on attempt {retry_count}: {e}")

        # If still failing, skip this sampling round
        if not success:
            print(f"[Error] All tables failed after {max_retries} retries in sample {sample_index}. Skipping this sample.")
            continue

        # Add successful results from this round to total samples
        all_samples.extend(all_table)

    # Merge all sampling results and simplify the structure
    try:
        merged_schema = merge_table_schemas(all_samples)
        simplified_schema = simplify_table_series(merged_schema)
        return list(simplified_schema.keys()), simplified_schema, sample_history
    except Exception as e:
        traceback.print_exc()
        print(f"[Error] Failed to merge table schemas from {len(all_samples)} samples: {e}")
        return [], {}, sample_history

def SL_workflow(Question_id, Question, db_id, Tool_model,
                model="deepseek-chat", 
                temperature=1.2, max_retries=10, max_token=50000,
                use_single_table: bool = False,db_type="snow",
                check_columns: bool = False,
                all_use_min:bool = False# Whether to perform simplification in all cases?
                ):
    """
    Main workflow function: Generates multiple SQLs based on the question and database structure, then extracts table schema information.
    
    Parameters:
        Question_id: Unique identifier for the question
        Question: Natural language question text
        db_id: Database name
        model: Name of the LLM model used
        temperature: Controls generation diversity
        max_retries: Maximum number of retry attempts
        max_token: Maximum token length limit
        use_single_table: Whether to handle single-table cases
        check_columns: Whether to verify column information
    """
    # Get all table names from the database
    if db_type=="snow":
        table_list = get_table_mess_snow(db_id)
    elif db_type=="sqlite":
        table_list = get_table_sqlite(db_id)
    else:
        table_list=get_table_mess_bigquery(db_id)
    # If it is a single-table database and use_single_table is disabled, skip subsequent processes
    if len(table_list) == 1 and not use_single_table:
        print(f"[Info] Database {db_id} has only one table, skipping SQL/LLM due to use_single_table=False")
        return table_list, {table_list[0]: []}, {0: {table_list[0]: []}}   # Return empty schema and sample_history

    # Get table structure information (JSON format)
    table_mess = M_Schema(db_id=db_id, SL=table_list, db_type=db_type, Level="table")
    #print(db_type,"\n",table_mess)
    # Get table structure DDL information
    if db_type=="snow" or db_type=="bigquery":
        table_mess_ddl = generate_ddl_from_json(db_id,db_type=db_type)
    elif db_type=="sqlite":
        table_mess_ddl = get_tables_ddl_sqlite(db_id,db_type=db_type)

    # Calculate token count for JSON schema
    len_table_mess = get_token_count(table_mess)

    # If JSON schema exceeds token limit
    if len_table_mess > max_token:
        print(f"[Info] Schema(token count: {len_table_mess}) exceeds max_token ({max_token}). Checking DDL schema as an alternative.")

        # Calculate token count for DDL schema
        len_table_mess_ddl = get_token_count(table_mess_ddl)

        # If DDL also exceeds token limit, switch to the old workflow
        if len_table_mess_ddl > max_token//2:
            print(f"[Info] DDL schema(token count: {len_table_mess_ddl}) also exceeds max_token. Switching to old workflow.")
            table_list_old,_,_=SL_workflow_old(
                Question_id, Question, db_id,
                model=model,
                Tool_model=Tool_model,
                temperature=temperature,
                max_retries=max_retries,
                db_type=db_type
            )
            print("[Info] Simplify again based on table-level schema links.")
            return SL_workflow_min(
                Question_id, Question, db_id,table_list_old,
                model=model,
                Tool_model=Tool_model,
                temperature=temperature,
                max_retries=max_retries,
                db_type=db_type
            )
        else:
            # Use DDL schema as an alternative to the original schema
            print(f"[Info] Using DDL schema (token count: {len_table_mess_ddl}) as it is within the limit.")
            table_mess = table_mess_ddl

    # Construct Prompt input content
    Prompt = get_prompt_SQL4(Question, table_mess, str(table_list),db_type=db_type)

    # Store results for each sampling round
    all_samples = []
    # Record structure returned by each sampling
    sample_history = {}

    # Perform 3 rounds of sampling
    for sample_index in range(3):
        success = False
        retry_count = 0
        all_table = []

        # Attempt up to max_retries times per round
        while not success and retry_count < max_retries:
            try:
                start_time = time.time()
                # print("Prompt:",Prompt)
                # Call LLM to get output        
                input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                    messages=[{"role": "user", "content": Prompt}],
                    model=model,
                    temperature=temperature
                )
                print("LLM_return:",LLM_return)
                log_llm_io(model_name=model, prompt=Prompt, output=LLM_return, think=Thinking, qid=Question_id)

                end_time = time.time()
                elapsed_time1 = end_time - start_time

                # Extract SQL statement
                SQL = extract_sql_code(LLM_return)

                start_time = time.time()

                # Execute SQL to get structure information
                if db_type == "snow" or db_type=="bigquery":
                    table_x = Get_SL_func_snow(SQL=SQL, db_name=db_id,model=Tool_model, check_columns=check_columns)
                elif db_type == "sqlite":
                    table_x = Get_SL_func_sqlite(SQL=SQL, db_name=db_id, model=Tool_model,check_columns=True)

                end_time = time.time()
                elapsed_time2 = end_time - start_time

                # Record the status of this run
                logger_status.log(
                    question_id=Question_id,
                    step=elapsed_time1 + elapsed_time2,
                    if_in_fix=model,
                    input_token_count=input_token_count,
                    output_token_count=output_token_count,
                    status=None
                )

                # If successful, record results and exit loop
                all_table.append(table_x)
                sample_history[sample_index] = table_x
                success = True

            except Exception as e:
                retry_count += 1
                traceback.print_exc()
                print(f"[Warning] Error processing all tables on attempt {retry_count}: {e}")

        # If still failing, skip this sampling round
        if not success:
            print(f"[Error] All tables failed after {max_retries} retries in sample {sample_index}. Skipping this sample.")
            continue

        # Add successful results from this round to total samples
        all_samples.extend(all_table)

    # Merge all sampling results and simplify the structure
    try:
        merged_schema = merge_table_schemas(all_samples)
        print("all_samples:",all_samples)
        print("merged_schema:",merged_schema)
        simplified_schema = simplify_table_series(merged_schema)
        if not all_use_min:# Do not simplify
            return list(simplified_schema.keys()), simplified_schema, sample_history
        else:# Perform secondary simplification in all cases
            return SL_workflow_min(
                Question_id, Question, db_id,list(simplified_schema.keys()),
                model=model,
                Tool_model=Tool_model,
                temperature=temperature,
                max_retries=max_retries,
                db_type=db_type
            )
    except Exception as e:
        traceback.print_exc()
        print(f"[Error] Failed to merge table schemas from {len(all_samples)} samples: {e}")
        return [], {}, sample_history


if __name__ == "__main__":
    # --- Modification 1: Add command line argument parsing ---
    parser = argparse.ArgumentParser(description="Run SL Workflow")
    parser.add_argument('--input', '-i', required=True, help="Input file path (.jsonl)")
    parser.add_argument('--output', '-o', required=True, help="Output file path (.json)")
    parser.add_argument('--model', '-m', default="deepseek-chat", help="Model name")
    parser.add_argument('--Tool_model', '-Tm', default="deepseek-chat", help="Model name")
    args = parser.parse_args()

    input_file_path = args.input
    output_file_path = args.output
    model_name = args.model  # For subsequent logic or SL_workflow usage
    Tool_model=args.Tool_model
    # --- Modification 2: Dynamically set Log path (LOG folder in the same directory as this script) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "LOG")
    os.makedirs(log_dir, exist_ok=True)  # Create if it does not exist
    log_file_path = os.path.join(log_dir, "V3_SL.jsonl")

    # --- Original logic (only modified log_file_path variable reference) ---
    logger_status = JsonLogger(log_file_path=log_file_path)
    MAX_TOKEN = 65536
    all_results = []
    processed_ids = set()
    need_list = []
    
    filter_by_need_list = bool(need_list)

    try:
        with open(output_file_path, 'r', encoding='utf-8') as outfile:
            existing_data = json.load(outfile)
            all_results.extend(existing_data)
            processed_ids.update(item['instance_id'] for item in existing_data)
            print(f"Loaded {len(existing_data)} existing records from '{output_file_path}'")
    except FileNotFoundError:
        print(f"No existing output file found at '{output_file_path}', starting fresh.")
    except json.JSONDecodeError:
        print(f"Invalid JSON in output file '{output_file_path}', starting fresh.")
        all_results = []
        processed_ids = set()

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                item = json.loads(line.strip())
                instance_id = item.get('instance_id', '')

                if filter_by_need_list and instance_id not in need_list:
                    print(f"Skipping item {line_num} (instance_id: {instance_id}), not in need_list")
                    continue

                if instance_id in processed_ids:
                    print(f"Skipping already processed item {line_num} (instance_id: {instance_id})")
                    continue

                print(f"\nProcessing item {line_num} (instance_id: {instance_id})...")

                question = item.get('question', '')
                db_name = item.get('db_id') or item.get('db') or ''
                evidence = item.get('evidence', '')

                if evidence:
                    user_input = f"[Evidence]\n{evidence}\n[Question]\n{question}\n"
                else:
                    user_input = f"[Question]\n{question}\n"
                
                db_type = detect_db_type(instance_id)
                
                # Note: If SL_workflow requires the model parameter, pass model=model_name here
                table, col, sample_history = SL_workflow(
                    Question_id=instance_id, 
                    Question=user_input, 
                    model=model_name,
                    db_id=db_name, 
                    max_token=MAX_TOKEN, 
                    all_use_min=True, 
                    db_type=db_type,
                    Tool_model=Tool_model
                )
                
                if not table and not col:
                    print(f"  -> Skipping save for item {line_num} (instance_id: {instance_id}), empty table/col")
                    continue
                
                output_item = item.copy()
                output_item['table'] = table
                output_item['col'] = col
                output_item["sample_history"] = sample_history

                all_results.append(output_item)
                processed_ids.add(instance_id)

                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(all_results, outfile, indent=4, ensure_ascii=False)

                print(f"  -> Saved {len(all_results)} item(s) to '{output_file_path}'")

        print(f"\nProcessing finished successfully. Final output is in '{output_file_path}'")

    except json.JSONDecodeError as e:
        print(f"\nError decoding JSON from a line in the input file: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"\nAn unexpected error occurred: {e}")
