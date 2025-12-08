import os
import re
import sys
import time
import json
import sqlite3
from typing import List, Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut

# Snowflake
import snowflake.connector
from snowflake.connector.errors import DatabaseError, ProgrammingError

# BigQuery
from google.oauth2 import service_account
from google.cloud import bigquery

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mytoken.deepseek_tokenizer import *
from LLM.LLM_OUT import LLM_output
from utils.Prompt import TOOL_LLM
from utils.DBsetup.Get_DB import read_db_config

# Import database information
sqlite_DB_dir, snow_DB_dir, bigquery_DB_dir, snow_auth, Credentials_Path = read_db_config()
default_credentials = json.load(open(snow_auth, 'r')) if snow_auth and os.path.exists(snow_auth) else {}

# Set maximum display rows to 20
pd.set_option('display.max_rows', 20)
# Set maximum display columns to 10
pd.set_option('display.max_columns', 10)

SQL_prompt='''
You are an agent specialized in completing repetitive code. Your job is to:

- Complete the missing UNION ALL CTEs in the SQL based on the comments in the SQL (and follow the same simple syntax rules!).
- You cannot be lazy; output the full amount of code needed without any omissions or ellipses.
- Don't try to be clever. Snowflake's syntax prohibits the use of wildcards, so you have to write it line by line!
- Also, be sure to handle date issues carefully—don’t make mistakes like 2/30!
- Currently, only when the current SQL provides merely a CTE, after you complete the CTE, you need to add a sentence "SELECT * FROM ctename LIMIT 1" to obtain an executable SQL. Otherwise, you only need to complete the CTE!
## SQL to complete  
{SQL}

## Output format (strictly in markdown)  
```sql
Completed SQL
```

Please start the task now. If you do a good job, I will tip you $100!
'''

def detect_db_type(instance_id: str) -> str:
    assert isinstance(instance_id, str) and instance_id, "Input instance_id must be a non-empty string."
    #re: https://github.com/xlang-ai/Spider2/blob/main/spider2-lite/evaluation_suite/evaluate.py
    instance_id_lower = instance_id.lower()

    if instance_id_lower.startswith(('bq', 'ga')):
        return "bigquery"
    elif instance_id_lower.startswith('sf'):
        return "snow"
    elif instance_id_lower.startswith('local'):
        return "sqlite"
    else:
        assert False, f"Unknown or unsupported database type prefix for instance_id: '{instance_id}'"

def extract_sql_block(text):
    pattern = r"```sql\s+(.*?)\s+```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def clean_table_name(table_name):
    table_name = str(table_name)
    table_name = table_name.replace('"', '')
    table_name = re.sub(r'\d+', '', table_name)
    table_name = table_name.lower()
    return table_name

def _execute_snowflake_query_inner(query, credentials, fetch_results=True, timeout=200):
    """
    Internal execution function: runs in a separate process.
    """
    conn = None
    cursor = None
    try:
        # Establish an independent connection within the child process
        conn = snowflake.connector.connect(
            user=credentials["user"],
            password=credentials["password"],
            account=credentials["account"],
            role=credentials["role"],
            warehouse=credentials["warehouse"],
        )
        cursor = conn.cursor()
        
        start_time = time.time()
        cursor.execute(query)

        if fetch_results:
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            if results:
                df = pd.DataFrame(results, columns=columns)
                # Simple formatting for DataFrame display
                df_str = df.to_string(index=True, show_dimensions=True, max_rows=20)
                execution_time = time.time() - start_time
                df_str_no_empty_lines = re.sub(r'\n\s*\n', '\n', df_str)
                
                return 0, truncate_text_by_tokens(df_str_no_empty_lines) + f"\nQuery Time: {execution_time:.2f} s"
            else:
                return 0, '[]'
        else:
            conn.commit()
            return 0, None

    except snowflake.connector.errors.ProgrammingError as pe:
        error_str = str(pe)
        if "000630" in error_str or "timed out" in error_str.lower():
            return 3, f"Timeout: {error_str}"
        return 1, f"Snowflake Programming Error: {pe}"
        
    except snowflake.connector.errors.DatabaseError as de:
        error_str = str(de)
        if "000630" in error_str or "timed out" in error_str.lower():
            return 3, f"Timeout: {error_str}"
        return 2, f"Snowflake Database Error: {de}"
        
    except Exception as e:
        return 3, f"Unknown Error: {e}"
    finally:
        try:
            if cursor: cursor.close()
            if conn: conn.close()
        except:
            pass

def _execute_sqlite_query_inner(query, db_path, fetch_results=True):
    
    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        start_time = time.perf_counter()

        cursor.execute(query)

        if fetch_results:
            results = cursor.fetchall()
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            if results:
                columns = [desc[0] for desc in cursor.description]
                results_pd = pd.DataFrame(results, columns=columns)
                df_str = results_pd.to_string(index=True, show_dimensions=True, max_rows=10)

                # Remove empty lines
                lines = df_str.splitlines()
                non_empty_lines = [line for line in lines if line.strip()]
                cleaned_df_str = '\n'.join(non_empty_lines)

                return 0, cleaned_df_str + f"\nQuery Time: {execution_time:.4f} s"
            else:
                return 0, f"[]\n\nQuery Time: {execution_time:.4f} s"
        else:
            conn.commit()
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return 0, f"Operation successful.\nExecution Time: {execution_time:.4f} s"

    except sqlite3.ProgrammingError as pe:
        return 1, f"SQLite Programming Error: {pe}"
    except sqlite3.DatabaseError as de:
        return 2, f"SQLite Database Error: {de}"
    except Exception as e:
        return 3, f"Unknown Error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def execute_sqlite_query(query, db_path, fetch_results=True):
    try:
        return func_timeout(30, _execute_sqlite_query_inner, args=(query, db_path, fetch_results))
    except FunctionTimedOut:
        # If the first 30 seconds are exceeded, try again with another 30-second grace period.
        try:
            return func_timeout(30, _execute_sqlite_query_inner, args=(query, db_path, fetch_results))
        except FunctionTimedOut:
            return 2, "SQLite Database Error: execution exceeded 30 seconds."

def execute_snowflake_query(query, credentials, db_id, fetch_results=True, timeout=200):
    """
    ⚠️⚠️⚠️ WARNING/警告 ⚠️⚠️⚠️
    !!!Using this function in real-world scenarios is a rather expensive and risky endeavor; this paper is constrained by the latency of the shared database and is solely aimed at maximizing the score.!!!
    !!!在真实场景中使用此函数是个相当昂贵并且冒险的行为，本文受限于共享数据库的延迟并且仅为了分数最大化!!!
    
    Aggressive concurrent query execution:
    1. Run 5 threads in parallel; use the result from the first to finish.
    2. Return immediately upon receiving a valid result, without waiting for other threads (using shutdown(wait=False)).
    3. Cost-agnostic mode: background threads are allowed to run to completion or terminate on their own.
    """
    max_retries = 10
    concurrency = 1
    
    for attempt in range(max_retries):
        pool = None
        try:
            # Start multiprocessing pool
            pool = multiprocessing.Pool(processes=concurrency)
            async_results = []
            
            # Submit tasks
            for _ in range(concurrency):
                # Note: apply_async is non-blocking
                res = pool.apply_async(
                    _execute_snowflake_query_inner, 
                    args=(query, credentials, fetch_results, timeout)
                )
                async_results.append(res)
            
            # Polling for results
            # We need to get the first result and kill the pool, so we can't use simple pool.map
            loop_start = time.time()
            finished_count = 0
            
            while True:
                any_success = False
                
                # Check status of all tasks
                for i, res in enumerate(async_results):
                    if res.ready():
                        try:
                            # The timeout here is for getting results, not query timeout
                            code, msg = res.get(timeout=0.1)
                            
                            # Judgment logic:
                            # code != 3 (not timeout) -> considered a valid result (whether success 0 or error 1,2), adopt it directly
                            if code != 3:
                                # 【Core Nuke】Found valid result, immediately terminate process pool
                                pool.terminate() 
                                pool.join() # Wait for OS to clean up zombie processes
                                return code, msg
                            
                            # If timeout (3), mark this task as abandoned, continue waiting for others
                            # To avoid duplicate checks, can remove it from list, or simply count
                            # Simple handling here: if result is 3, do nothing, wait for loop to end to judge if all failed
                            
                        except Exception as e:
                            # Process communication exception, ignore, check next one
                            pass
                
                # Count how many tasks have completed (even if failed/timed out)
                finished_count = sum(1 for res in async_results if res.ready())
                
                # If all concurrent tasks have finished, and code reaches here, it means none had code != 3
                # Then this batch all timed out/failed
                if finished_count == concurrency:
                    print(f"Batch {attempt + 1} all timed out or failed. Retrying...")
                    pool.terminate()
                    pool.join()
                    break # Break out of while, enter outer for for next retry
                
                # Avoid CPU busy-waiting
                time.sleep(0.1)
                
                # Additional total timeout protection (prevent processes from hanging)
                if time.time() - loop_start > timeout + 10:
                    print(f"Batch {attempt + 1} hard timeout limit reached.")
                    pool.terminate()
                    pool.join()
                    break

        except Exception as e:
            print(f"Critical error in batch execution: {e}")
            if pool:
                pool.terminate()
                pool.join()
        finally:
            # Fallback: ensure pool is definitely closed
            if pool:
                # close() waits for tasks to complete, terminate() kills immediately. Here use terminate to ensure no residue
                try:
                    pool.terminate() 
                    pool.join()
                except:
                    pass

    return 3, f"Execution timed out after {max_retries} attempts."

def _execute_bigquery_query_inner(query, credentials_path, fetch_results=True):

    try:
        t0 = time.time()
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = bigquery.Client(credentials=credentials)
        t1 = time.time()
        print(f"Connection time: {t1 - t0:.2f} S")

        start_time = time.time()
        query_job = client.query(query)

        if fetch_results:
            results = query_job.result()
            df = pd.DataFrame([dict(row.items()) for row in results])
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"BigQuery job complete. Cache hit: {query_job.cache_hit}. Data billed: {(query_job.total_bytes_billed or 0) / 1024 / 1024:.2f} MB")
            if not df.empty:
                df_str = df.to_string(index=True, show_dimensions=True, max_rows=20)
                df_str_no_empty_lines = re.sub(r'\n\s*\n', '\n', df_str)
                return 0, truncate_text_by_tokens(df_str_no_empty_lines) + f"\nQuery Time: {execution_time:.2f} s"
            else:
                return 0, '[]'
        else:
            query_job.result()
            return 0, None
    except Exception as e:
        return 3, f"BigQuery programming Error: {e}"

def execute_bigquery_query(query, credentials_path, fetch_results=True, timeout=200):
    try:
        return func_timeout(timeout, _execute_bigquery_query_inner,
                            args=(query, credentials_path, fetch_results))
    except FunctionTimedOut:
        return 3, f"Execution timed out after {timeout} seconds."

def db_interface(db_type, query, conn_info, fetch_results=True):
    """
    Unified database interface that selects the appropriate execution function based on the database type.
    Args:
        db_type (str): The database type, supports 'sqlite', 'snowflake', or 'bigquery'.
        query (str): The SQL query to be executed.
        conn_info (str or dict): 
            - For sqlite, this can be the database filename or the full path.
            - For snowflake, this should be the database ID.
            - For BigQuery, this is not used as the JSON credential path is handled globally.
        fetch_results (bool): Whether to fetch query results (default is True).
    Returns:
        tuple: (status_code, query_result_or_error_message)
    """
    db_type = db_type.lower()
    if db_type == 'sqlite':
        # Base path for SQLite DBs
        if not conn_info.endswith(".sqlite"):
            # If only the database name is provided, construct the path automatically.
            conn_info = os.path.join(sqlite_DB_dir, conn_info, f"{conn_info}.sqlite")
        return execute_sqlite_query(query, conn_info, fetch_results)
    
    if db_type == "snow":#Snowflake
        return execute_snowflake_query(query, credentials=default_credentials, db_id=conn_info)
    
    if db_type == "bigquery":
        return execute_bigquery_query(query, credentials_path=Credentials_Path)
    
    return 3, "Support for other database types is not yet implemented."

def SQL_completion(text, db_type="snow"):
    """
    In Snowflake, there are some SQL statements with a length exceeding 60,000 tokens. 
    Normally, this type of data should not be handled by the Snowflake database, or alternative table operation permissions should be granted.
    Such tasks do not seem to exist in spider2.0-lite.
    """
    if db_type != "snow":
        return 1, text
        
    # Check for specific keywords that trigger the completion logic 
    # There is currently no better judgment logic and method
    if ("union" in text.lower() and "repeated" in text.lower()) or \
       ("union" in text.lower() and "note" in text.lower()) or \
       ("repeated" in text.lower() and "note" in text.lower()) or \
       ("union" in text.lower() and "repeat" in text.lower()) or \
       ("union" in text.lower() and "table" in text.lower() and "/*" in text.lower()):
        
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                SQL_mess = [{"role": "user", "content": SQL_prompt.format(SQL=text)}]
                input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                    messages=SQL_mess,
                    model=TOOL_LLM,
                    temperature=0,
                    # enable_thinking=False
                )
                SQL = extract_sql_block(text=LLM_return)
                if SQL:
                    return 0, SQL
                else:
                    print(f"⚠️ Attempt {attempt} failed: Failed to extract SQL.")
            except Exception as e:
                print(f"⚠️ An exception occurred on attempt {attempt}: {e}")
        
        print("❌ Max retries reached. Returning the original text.")
        return 1, text
    else:
        return 1, text
    
def M_Schema_sqlite(SL, db_id, level='table'):
    """
    Generates a formatted database schema string based on the given database ID (db_id),
    table/column selection list (SL), and level.
    """
    if db_id is None:
        raise ValueError("The db_id parameter must be provided.")

    # Assume sqlite_DB_dir is a defined path variable
    db_dir = os.path.join(sqlite_DB_dir, db_id)
    json_path = os.path.join(db_dir, f"{db_id}_M-Schema.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Schema file not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load schema file: {str(e)}")

    all_tables_info = schema_data.get(db_id, {})
    all_foreign_keys = schema_data.get("foreign_keys", {})
    table_name_map = {name.lower(): name for name in all_tables_info.keys()}

    if SL is None:
        SL = list(all_tables_info.keys())
        level = 'table'

    tables_to_process = []

    if level == 'table':
        if not isinstance(SL, list):
            raise TypeError("When level='table', SL must be a list.")

        lower_sl_tables = [t.lower() for t in SL]
        tables_to_process = [table_name_map[t] for t in lower_sl_tables if t in table_name_map]

        missing_tables = set(lower_sl_tables) - set(table_name_map.keys())
        if missing_tables:
            print(f"Warning: Tables {missing_tables} do not exist in database '{db_id}'.")

    elif level == 'column':
        if not isinstance(SL, dict):
            raise TypeError("When level='column', SL must be a dictionary (dict).")

        lower_sl_tables = [t.lower() for t in SL.keys()]
        tables_to_process = [table_name_map[t] for t in lower_sl_tables if t in table_name_map]

        missing_tables = set(lower_sl_tables) - set(table_name_map.keys())
        if missing_tables:
            print(f"Warning: Tables {missing_tables} do not exist in database '{db_id}'.")
    else:
        raise ValueError(f"Invalid level parameter: '{level}'. Only 'table' or 'column' is supported.")

    lines = [f"Note that the 'Examples' are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns or values to use.\n[DB_ID] {db_id}"]

    for original_table_name in tables_to_process:
        columns_data = all_tables_info.get(original_table_name, [])
        lines.append(f"# Table: {original_table_name}")
        lines.append("[")

        cols_to_render = []
        if level == 'table':
            cols_to_render = columns_data
        elif level == 'column':
            requested_cols = []
            for sl_key, sl_val in SL.items():
                if sl_key.lower() == original_table_name.lower():
                    requested_cols = [c.lower() for c in sl_val]
                    break

            table_col_map = {col[0].lower(): col for col in columns_data}
            for req_col_lower in requested_cols:
                if req_col_lower in table_col_map:
                    cols_to_render.append(table_col_map[req_col_lower])
                else:
                    print(f"Warning: Column '{req_col_lower}' does not exist in table '{original_table_name}', skipping.")

        col_lines = []
        for col_list in cols_to_render:
            col_name, pk_info, col_type, col_desc, col_examples = col_list
            col_parts = [f"{col_name}: {col_type}"]
            if pk_info == "Primary Key":
                col_parts.append("Primary Key")
            if col_desc:
                col_parts.append(col_desc)
            if col_examples:
                col_parts.append(f"Examples: [{col_examples}]")

            col_str = f"({', '.join(col_parts)})"
            col_lines.append(col_str)

        for i, line in enumerate(col_lines):
            lines.append(line + ("," if i < len(col_lines) - 1 else ""))
        lines.append("]")

    if len(tables_to_process) > 1:
        relevant_fks = []
        tables_to_process_lower = {t.lower() for t in tables_to_process}
        for source_col, target_col in all_foreign_keys.items():
            source_table = source_col.split('.')[0].lower()
            target_table = target_col.split('.')[0].lower()
            if source_table in tables_to_process_lower and target_table in tables_to_process_lower:
                relevant_fks.append(f"{source_col} = {target_col}")

        if relevant_fks:
            lines.append("[Foreign keys]")
            lines.extend(relevant_fks)

    return "\n".join(lines)

def M_Schema_bigquery(db_id, SL=None) -> str:

    if SL is None:
        SL = []

    def _simplify_list_series(table_list):
        seen_series = set()
        result = []
        for table_name in table_list:
            series_name = re.sub(r'_\d{4}(?:_\d+yr)?$', '', table_name.lower())
            series_name = re.sub(r'\d+', '', series_name)
            if series_name not in seen_series:
                result.append(table_name)
                seen_series.add(series_name)
        return result
    
    SL = _simplify_list_series(SL)
    try:
        if not os.path.isdir(bigquery_DB_dir):
            raise FileNotFoundError(f"Base directory not found: {bigquery_DB_dir}")

        correct_cased_dirname = None
        for dirname in os.listdir(bigquery_DB_dir):
            if dirname.lower() == db_id.lower():
                if os.path.isdir(os.path.join(bigquery_DB_dir, dirname)):
                    correct_cased_dirname = dirname
                    break

        if not correct_cased_dirname:
            raise FileNotFoundError(
                f"Directory for db_id '{db_id}' not found (case-insensitive search) in '{bigquery_DB_dir}'."
            )

        db_dir = os.path.join(bigquery_DB_dir, correct_cased_dirname)
        json_path = os.path.join(db_dir, f"{correct_cased_dirname}_M-Schema.json")

    except NameError:
        raise NameError("Global variable 'bigquery_DB_dir' is not defined.")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Database schema file not found for db_id '{db_id}' at {json_path}.")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse JSON from {json_path}: {str(e)}")
    if not all_data:
        raise ValueError(f"JSON file {json_path} is empty or invalid.")

    def _format_table_details(full_table_name, column_details, table_description=None) -> list[str]:
        table_lines = [f"# Table: {full_table_name}", "["]
        if not column_details:
            table_lines.append(f"  (Detailed column information not found for table)")
        else:
            formatted_columns = []
            for col_info in column_details:
                if len(col_info) < 4: continue
                col_name, col_type, col_desc, col_examples = col_info
                column_parts = [f"{col_name}: {col_type}"]
                if col_desc and col_desc.strip(): column_parts.append(col_desc.strip())
                if col_examples:
                    example_text = col_examples.replace("examples:", "").strip()
                    if example_text: column_parts.append(f"Examples: {example_text}")
                parts_str = f"({', '.join(column_parts)})"
                formatted_columns.append(parts_str)
            for i, line in enumerate(formatted_columns):
                table_lines.append("  " + line + ("," if i < len(formatted_columns) - 1 else ""))
        table_lines.append("]")
        if table_description:
            table_lines.append(f"# Table Description: {table_description}")
        return table_lines
    
    lines = [f"【TASK_ID】{db_id}"]

    if not SL:
        for top_level_key, db_content in all_data.items():
            if not isinstance(db_content, dict): continue
            
            for dataset_name, dataset_content in db_content.items():
                if not isinstance(dataset_content, dict): continue
                
                desc_summary_dict = dataset_content.get("table_description_summary", {})
                
                for table_key_original, column_details in dataset_content.items():
                    if table_key_original in ["table_Information", "table_description_summary"]:
                        continue
                    
                    full_table_name = f"{top_level_key}.{table_key_original}"
                    table_desc = desc_summary_dict.get(table_key_original)
                    
                    table_formatted_lines = _format_table_details(full_table_name, column_details, table_desc)
                    lines.extend(table_formatted_lines)
    else:
        full_dataset_map = {} 
        full_table_map = {}  
        
        for top_level_key, db_content in all_data.items():
            if not isinstance(db_content, dict): continue
            for dataset_original, dataset_content in db_content.items():
                if not isinstance(dataset_content, dict): continue
                
                full_dataset_map[dataset_original.lower()] = dataset_original
                
                for table_key_original in dataset_content.keys():
                    if table_key_original not in ["table_Information", "table_description_summary"]:
                        # a short table id is like 'bbc_news.fulltext'
                        short_table_id_lower = table_key_original.lower()
                        full_table_map[short_table_id_lower] = table_key_original
        
        displayed_surrogate_descriptions = set()

        for full_table_name_input in SL:
            full_table_name_lower = full_table_name_input.lower()
            parts = full_table_name_lower.split('.')
            if len(parts) < 3:
                print(f"Warning: Skipping invalid table name format: {full_table_name_input}")
                continue
            current_top_level_key_lower, dataset_name_lower, table_name_part_lower = parts[0], parts[1], '.'.join(parts[2:])
            original_top_level_key = next((k for k in all_data.keys() if k.lower() == current_top_level_key_lower), None)
            if not original_top_level_key:
                print(f"Warning: Table '{full_table_name_input}' top-level key '{current_top_level_key_lower}' not found in JSON. Skipping.")
                continue

            db_content = all_data[original_top_level_key]
            
            table_id_short_lower = f"{dataset_name_lower}.{table_name_part_lower}"
            key_for_lookup, surrogate_key, is_found = None, None, False

            if table_id_short_lower in full_table_map:
                key_for_lookup = full_table_map[table_id_short_lower]
                is_found = True
                surrogate_key = key_for_lookup

            if not is_found:
                dataset_name_original = full_dataset_map.get(dataset_name_lower)
                if dataset_name_original and dataset_name_original in db_content:
                    dataset_content = db_content[dataset_name_original]
                    table_info = dataset_content.get("table_Information", {})
                    for surrogate, similar_tables in table_info.items():
                        tables = []
                        if isinstance(similar_tables, dict): tables = similar_tables.get("similar_tables", [])
                        elif isinstance(similar_tables, list): tables = similar_tables
                        if table_id_short_lower in [t.lower() for t in tables]:
                            key_for_lookup, surrogate_key, is_found = surrogate, surrogate, True
                            break
            
            if not is_found:
                print(f"Warning: Table '{full_table_name_input}' not found. Skipping.")
                continue

            display_full_table_name = f"{original_top_level_key}.{key_for_lookup}"
            lookup_dataset_name = key_for_lookup.split('.')[0]
            
            dataset_data = db_content.get(lookup_dataset_name, {})
            column_details = dataset_data.get(key_for_lookup, [])
            
            table_desc = None
            if surrogate_key and surrogate_key not in displayed_surrogate_descriptions:
                desc_dataset_name = surrogate_key.split('.')[0]
                desc_summary_dict = db_content.get(desc_dataset_name, {}).get("table_description_summary", {})
                table_desc = desc_summary_dict.get(surrogate_key)
                if table_desc:
                    displayed_surrogate_descriptions.add(surrogate_key)

            table_formatted_lines = _format_table_details(display_full_table_name, column_details, table_desc)
            lines.extend(table_formatted_lines)

    return "\n".join(lines)

def M_Schema(db_id, SL=None, db_type="snow", Level="table") -> str:

    if db_type=="sqlite":
        return M_Schema_sqlite(db_id=db_id, SL=SL)
    if db_type=="bigquery":
        return M_Schema_bigquery(db_id=db_id, SL=SL)

    # If SL is None, initialize it as an empty list for later processing
    if SL is None:
        SL = []

    def _simplify_list_series(table_list):
        seen_series = set()  # Use a set for efficient tracking
        result = []
        for table_name in table_list:
            series_name = re.sub(r'\d+', '', table_name.lower())
            if series_name not in seen_series:
                result.append(table_name)
                seen_series.add(series_name)
        return result
    SL=_simplify_list_series(SL)
    
    # --- 1. Loading and Initialization ---
    try:
        # Ensure the global variable exists
        db_dir = os.path.join(snow_DB_dir, db_id)
    except NameError:
        raise NameError("Global variable 'snow_DB_dir' is not defined. Please define it before calling this function.")

    json_path_options = [os.path.join(db_dir, f"{db_id}_M-Schema.json")]
    json_path = next((path for path in json_path_options if os.path.exists(path)), None)

    if not json_path:
        raise FileNotFoundError(f"Database schema file not found for db_id '{db_id}' in expected locations.")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_schemas_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse JSON from {json_path}: {str(e)}")
        
    # --- Helper function: Format details for a single table ---
    def _format_table_details(full_table_name, column_details, table_description=None) -> list[str]:
        """Generate a list of formatted strings for a single table."""
        table_lines = [f"# Table: {full_table_name}", "["]
        
        if not column_details:
            table_lines.append(f"  (Detailed column information not found for table)")
        else:
            formatted_columns = []
            for col_info in column_details:
                if len(col_info) < 4: continue
                col_name, col_type, col_desc, col_examples = col_info
                
                column_parts = [f"{col_name}: {col_type}"]
                if col_desc and col_desc.strip():
                    column_parts.append(col_desc.strip())
                if col_examples:
                    example_text = col_examples.replace("examples:", "").strip()
                    if example_text:
                        column_parts.append(f"Examples: {example_text}")
                
                parts_str = f"({', '.join(column_parts)})"
                formatted_columns.append(parts_str)
            
            for i, line in enumerate(formatted_columns):
                table_lines.append("  " + line + ("," if i < len(formatted_columns) - 1 else ""))
        
        table_lines.append("]")
        
        if table_description:
            table_lines.append(f"# Table Description: {table_description}")
            
        return table_lines

    # --- 2. Main Logic: Behavior depends on whether SL is empty ---
    lines = [f"[DB_ID] {db_id}", "[Schema]"]

    # Branch A: SL is empty, output the schema for the entire database
    if not SL:
        for schema_original, schema_content in all_schemas_data.items():
            if not isinstance(schema_content, dict): continue
            
            desc_summary_dict = schema_content.get("table_description_summary", {})
            
            # Iterate through all tables in the schema
            for table_key_original, column_details in schema_content.items():
                # Ignore metadata keys
                if table_key_original in ["table_Information", "table_description_summary"]:
                    continue
                
                full_table_name = f"{db_id}.{table_key_original}"
                table_desc = desc_summary_dict.get(table_key_original)
                
                table_formatted_lines = _format_table_details(full_table_name, column_details, table_desc)
                lines.extend(table_formatted_lines)
    
    # Branch B: SL is not empty, find and output the specified tables
    else:
        # Create case-insensitive lookup maps (only when needed)
        schema_map = {s.lower(): s for s in all_schemas_data.keys()}
        table_map = {}
        for schema_lower, schema_original in schema_map.items():
            schema_content = all_schemas_data[schema_original]
            if isinstance(schema_content, dict):
                for table_key_original in schema_content.keys():
                    if table_key_original not in ["table_Information", "table_description_summary"]:
                        table_map[table_key_original.lower()] = table_key_original
        
        displayed_surrogate_descriptions = set()

        for full_table_name_input in SL:
            full_table_name_lower = full_table_name_input.lower()
            parts = full_table_name_lower.split('.')
            if len(parts) < 3:
                print(f"Warning: Skipping invalid table name format: {full_table_name_input}")
                continue

            current_db_id_lower, schema_name_lower, table_name_part_lower = parts[0], parts[1], '.'.join(parts[2:])
            table_id_short_lower = f"{schema_name_lower}.{table_name_part_lower}"
            
            if current_db_id_lower != db_id.lower():
                print(f"Warning: Table '{full_table_name_input}' DB mismatch (expected '{db_id}'). Skipping.")
                continue
            
            # Find the real information for the table
            key_for_lookup = None
            surrogate_key = None
            is_found = False

            # Step 1: Attempt direct match
            if table_id_short_lower in table_map:
                key_for_lookup = table_map[table_id_short_lower]
                is_found = True
                surrogate_key = key_for_lookup  # Assume the directly found key is its own surrogate

            # Step 2: If direct match fails, check if it's a similar table of a surrogate
            if not is_found:
                schema_name_original = schema_map.get(schema_name_lower)
                if schema_name_original and schema_name_original in all_schemas_data:
                    table_info = all_schemas_data[schema_name_original].get("table_Information", {})
                    for surrogate, similar_tables in table_info.items():
                        # Compatible with both list and dict formats
                        if isinstance(similar_tables, dict):
                            tables = similar_tables.get("similar_tables", [])
                        elif isinstance(similar_tables, list):
                            tables = similar_tables
                        else:
                            tables = []

                        if table_id_short_lower in [t.lower() for t in tables]:
                            key_for_lookup = surrogate
                            surrogate_key = surrogate
                            is_found = True
                            break

            if not is_found:
                print(f"Warning: Table '{full_table_name_input}' not found directly or as a similar table. Skipping.")
                continue

            # Get and format the table details
            display_full_table_name = f"{db_id}.{key_for_lookup}"
            lookup_schema_name = key_for_lookup.split('.')[0]
            
            schema_data = all_schemas_data.get(lookup_schema_name, {})
            column_details = schema_data.get(key_for_lookup, [])
            
            table_desc = None
            if surrogate_key and surrogate_key not in displayed_surrogate_descriptions:
                desc_schema_name = surrogate_key.split('.')[0]
                desc_summary_dict = all_schemas_data.get(desc_schema_name, {}).get("table_description_summary", {})
                table_desc = desc_summary_dict.get(surrogate_key)
                if table_desc:
                    displayed_surrogate_descriptions.add(surrogate_key)

            table_formatted_lines = _format_table_details(display_full_table_name, column_details, table_desc)
            lines.extend(table_formatted_lines)

    return "\n".join(lines)

def generate_ddl_from_json(db_id, table_list=None, db_type="snow"):
    # Delegate to specific functions for sqlite or bigquery
    if db_type == "sqlite":
        return get_tables_ddl_sqlite(db_id, table_list)
    if db_type == "bigquery":
        return generate_ddl_from_json_bigquery(db_id, table_list)
    
    # Construct the path to the database directory
    db_dir = os.path.join(snow_DB_dir, db_id)

    # Define potential paths for the schema file
    json_path_options = [
        os.path.join(db_dir, f"{db_id}_M-Schema.json")
    ]
    # Find the first existing schema file from the options
    json_path = next((path for path in json_path_options if os.path.exists(path)), None)

    if not json_path:
        raise FileNotFoundError(f"Database schema file not found for db_id '{db_id}' in expected locations.")

    try:
        # Load the schema data from the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            all_schemas_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse JSON from {json_path}: {str(e)}")

    # --- Create a set of cleaned table names for robust matching ---
    cleaned_tables_to_include_set = None
    if table_list:
        # Apply the cleaning function to each table name in table_list and store it in a set
        cleaned_tables_to_include_set = {clean_table_name(t) for t in table_list}

    ddl_statements = []

    # Mapping from schema types to SQL types
    type_mapping = {
        "TEXT": "TEXT",
        "NUMBER": "INTEGER",
        "FLOAT": "REAL",
        "DATE": "DATE",
        "TIME": "TIME"
    }

    # Iterate through each schema in the loaded data
    for schema_name, schema_content in all_schemas_data.items():
        if not isinstance(schema_content, dict):
            continue

        table_summaries = schema_content.get("table_description_summary", {})

        # Iterate through each table within the schema
        for table_name, columns in schema_content.items():
            if not isinstance(columns, list):
                continue
            
            # --- Clean the table name from JSON for comparison ---
            if cleaned_tables_to_include_set:
                # 1. Construct the original full table name
                original_full_name = f"{db_id}.{table_name}"
                # 2. Clean the name for comparison
                cleaned_full_name_for_check = clean_table_name(original_full_name)
                # 3. Look for it in the cleaned set
                if cleaned_full_name_for_check not in cleaned_tables_to_include_set:
                    # If the cleaned name doesn't match, skip this table
                    continue
            
            # --- DDL generation logic remains unchanged, always using original names ---
            # Note: We use the original db_id and table_name here to generate the DDL
            fully_qualified_table_name_quoted = f'"{db_id}.{table_name}"'
            create_statement = f'CREATE TABLE {fully_qualified_table_name_quoted} (\n'
            
            column_definitions = []
            # Iterate through columns to define them
            for col_info in columns:
                if not isinstance(col_info, list) or len(col_info) < 2:
                    continue

                col_name = col_info[0]
                col_type = col_info[1]
                sql_type = type_mapping.get(col_type.upper(), col_type)
                
                col_def = f'    "{col_name}" {sql_type}'
                column_definitions.append(col_def)

            create_statement += ',\n'.join(column_definitions)
            create_statement += '\n);\n'
            
            # Add table summary as a comment if it exists
            summary_text = table_summaries.get(table_name)
            if summary_text:
                summary_comment = f"\n/*\n{summary_text.strip()}\n*/\n"
                create_statement += summary_comment

            ddl_statements.append(create_statement)

    # Join all generated DDL statements into a single string
    return "\n".join(ddl_statements)

def get_tables_ddl_sqlite(db_id: str, table_list: Optional[List[str]] = None) -> str:
    db_path = f"{sqlite_DB_dir}/{db_id}/{db_id}.sqlite"
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return ""

    ddl_statements = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            base_query = "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            params = []

            if table_list:
                placeholders = ', '.join(['?'] * len(table_list))
                query = f"{base_query} AND name IN ({placeholders});"
                params = table_list
            else:
                query = base_query + ";"

            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            ddl_statements = [row[0] for row in rows if row and row[0]]

    except sqlite3.Error as e:
        print(f"Error connecting to or querying database '{db_path}': {e}")
        return ""

    if not ddl_statements:
        if table_list:
             print(f"Could not find the specified tables in database '{db_path}': {table_list}.")
        else:
             print(f"No user-created tables found in database '{db_path}'.")
        return ""
    
    return ';\n\n'.join(ddl_statements) + ';'

def generate_ddl_from_json_bigquery(db_id, table_list=None):

    try:
        if not os.path.isdir(bigquery_DB_dir):
            raise FileNotFoundError(f"Base directory not found: {bigquery_DB_dir}")

        correct_cased_dirname = None
        for dirname in os.listdir(bigquery_DB_dir):
            if dirname.lower() == db_id.lower() and os.path.isdir(os.path.join(bigquery_DB_dir, dirname)):
                correct_cased_dirname = dirname
                break

        if not correct_cased_dirname:
            raise FileNotFoundError(
                f"Directory for db_id '{db_id}' not found (case-insensitive search) in '{bigquery_DB_dir}'."
            )

        db_dir = os.path.join(bigquery_DB_dir, correct_cased_dirname)
        json_path = os.path.join(db_dir, f"{correct_cased_dirname}_M-Schema.json")
    
    except NameError:
        raise NameError("Global variable 'bigquery_DB_dir' is not defined.")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Database schema file not found for db_id '{db_id}' at {json_path}.")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse JSON from {json_path}: {str(e)}")
    if not all_data:
        raise ValueError(f"JSON file {json_path} is empty or invalid.")
    cleaned_tables_to_include_set = None
    if table_list:
        cleaned_tables_to_include_set = {clean_table_name(t) for t in table_list}

    ddl_statements = []
    type_mapping = {
        "STRING": "STRING", "TEXT": "STRING",
        "NUMBER": "INT64", "INTEGER": "INT64",
        "FLOAT": "FLOAT64", "REAL": "FLOAT64",
        "DATE": "DATE", "TIME": "TIME",
        "TIMESTAMP": "TIMESTAMP", "DATETIME": "DATETIME",
        "BOOLEAN": "BOOL", "GEOGRAPHY": "GEOGRAPHY",
    }
    COMPLEX_TYPE_LENGTH_THRESHOLD = 50
    PLACEHOLDER_COMPLEX_TYPE = "COMPLEX_TYPE" #Nested types are uniformly referred to as COMPLEX_TYPE.

    for top_level_key, db_content in all_data.items():
        if not isinstance(db_content, dict):
            continue
        for dataset_name, dataset_content in db_content.items():
            if not isinstance(dataset_content, dict): continue

            table_summaries = dataset_content.get("table_description_summary", {})
            for table_key, columns in dataset_content.items():
                if not isinstance(columns, list): continue

                original_full_name = f"{top_level_key}.{table_key}"
                
                if cleaned_tables_to_include_set:
                    cleaned_full_name_for_check = clean_table_name(original_full_name)
                    if cleaned_full_name_for_check not in cleaned_tables_to_include_set:
                        continue
                
                fully_qualified_table_name_quoted = f'`{original_full_name}`'
                create_statement = f'CREATE TABLE {fully_qualified_table_name_quoted} (\n'
                
                column_definitions = []
                for col_info in columns:
                    if not isinstance(col_info, list) or len(col_info) < 2: continue

                    col_name = col_info[0]
                    col_type = col_info[1]
                    sql_type = "" 

                    if col_type and len(col_type) > COMPLEX_TYPE_LENGTH_THRESHOLD:
                        sql_type = PLACEHOLDER_COMPLEX_TYPE
                    else:
                        sql_type = type_mapping.get(col_type.upper(), col_type.upper()) if col_type else "UNKNOWN"

                    col_def = f'    `{col_name}` {sql_type}'
                    column_definitions.append(col_def)

                if not column_definitions:
                    continue

                create_statement += ',\n'.join(column_definitions)
                create_statement += '\n);\n'
                
                summary_text = table_summaries.get(table_key)
                if summary_text:
                    summary_comment = f"\n/*\n{summary_text.strip()}\n*/\n"
                    create_statement += summary_comment

                ddl_statements.append(create_statement)

    return "\n".join(ddl_statements)


if __name__ == "__main__":
#     query='''
# -- Revised: Filter boundaries by admin_level without subquery
# WITH filtered_boundaries AS (
#   SELECT 
#     t."osm_way_id",
#     TRY_TO_GEOGRAPHY(t."geometry") AS boundary_geom,
#     admin.value:value::STRING AS admin_level
#   FROM "GEO_OPENSTREETMAP"."GEO_OPENSTREETMAP"."PLANET_FEATURES" t,
#     LATERAL FLATTEN(t."all_tags") boundary,
#     LATERAL FLATTEN(t."all_tags") admin
#   WHERE t."feature_type" = 'multipolygons'
#     AND boundary.value:key::STRING = 'boundary'
#     AND boundary.value:value::STRING = 'administrative'
#     AND admin.value:key::STRING = 'admin_level'
#     AND admin.value:value::STRING IN ('6','7','8','9','10')
#     AND boundary_geom IS NOT NULL
# ),
# pois AS (
#   SELECT 
#     n."id" AS poi_id,
#     ST_MAKEPOINT(n."longitude", n."latitude") AS poi_geom
#   FROM "GEO_OPENSTREETMAP"."GEO_OPENSTREETMAP"."PLANET_NODES" n,
#     LATERAL FLATTEN(n."all_tags") ft
#   WHERE ft.value:key::STRING = 'amenity'
#     AND poi_geom IS NOT NULL
# )
# SELECT 
#   b."osm_way_id",
#   COUNT(DISTINCT p.poi_id) AS poi_count
# FROM filtered_boundaries b
# LEFT JOIN pois p 
#   ON ST_WITHIN(p.poi_geom, b.boundary_geom)
# GROUP BY b."osm_way_id"
# HAVING poi_count > 0
# LIMIT 5;
# '''
#     res=db_interface(db_type="snow",query=query,conn_info=default_credentials)
#     print(res[1])
    query='''
SELECT
*
FROM
  `bigquery-public-data.hacker_news.INFORMATION_SCHEMA.COLUMNS`
WHERE
  table_name = 'full'
ORDER BY
  ordinal_position;
--  Final score: 0.43349753694581283, Correct examples: 88, Total examples: 203
'''
    res=db_interface(db_type="bigquery",query=query,conn_info=Credentials_Path)
    print(res[1])