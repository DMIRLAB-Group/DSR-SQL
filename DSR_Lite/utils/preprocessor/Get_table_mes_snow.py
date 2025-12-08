import os
import sys
import re
import json
import time
import copy
import pickle
from datetime import datetime
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


import pandas as pd  # 用于数据处理
import snowflake.connector
from snowflake.connector.errors import ProgrammingError

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from LLM.LLM_OUT import LLM_output
from utils.extract_json import extract_and_parse_json
from utils.DBsetup.Get_DB import read_db_config

_, snow_DB_dir, _, SNOWFLAKE_CREDENTIALS, _=read_db_config()


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush() 

    def flush(self):
        for f in self.files:
            f.flush()


prompt='''
I have a set of database tables with the same structure but varying names. Please:  
1. Induce the naming rules of these tables, marking the variable parts with descriptive placeholders (e.g., `[REGION]`, `[VERSION]`, `[TYPE]`, etc.).  
2. Summarize the meaning or representation of these tables in a short paragraph.  
3. For each placeholder, explain its meaning and all value ranges appearing in this set of table names (e.g., "\[VERSION] ranges from R24 to R37, including R33P1").  

An example is as follows:
【Table_Name】
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R34": [
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R35",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R25",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R27",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R34",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R37",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R32",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R36",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R37",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R26",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R28",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R33P1",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R31",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R36",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R30",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R35",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R25",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R27",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R29",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R28",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R24",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG19_GDC_R29",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R24",
"TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_HG38_GDC_R26"
]
【Answer】
The naming convention for these tables with the same structure is:
`TCGA_VERSIONED.PER_SAMPLE_FILE_METADATA_[GENOME_REFERENCE]_GDC_[RELEASE_VERSION]`, where:  
- `[GENOME_REFERENCE]` denotes the genome reference version, including `HG19` or `HG38`.  
- `[RELEASE_VERSION]` represents the GDC data release version, formatted as `R[digit]` or `R[digit]P[patch number]`, currently including `R24` to `R37`, and the patched version `R33P1`. Among them, columns R30 to R37 have an additional "ICLR_ID" column.  
These tables represent the file metadata per sample for different versions of the human genome (HG19 or HG38) across various GDC data release versions.

Now, please handle the following problem:
【Table_Name】
{Table_Name}
【Column_Description】
{Column_Description}
【Extra_info】
{Extra_info}

## Output format(Markdown)
<Analysis Summary>Analysis Summary put thhere</Analysis Summary>
**return**
```json
{{
"Answer":"Please provide the answer here (refer to the example)!",
}}
```
'''


def remove_digits(s):
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r'\d', '', s)

def collect_single_database_info(db_path):
    """
    Collects complete hierarchical information for a single database at the specified path.

    This function directly processes a database directory, extracts table and column information 
    for all its schemas, applies table name correction logic, and finally returns 
    the information dictionary for that database.

    Directory structure should be:
    db_path/
    ├── [schema_name]/
    │   ├── [table_name_1].json
    │   └── [table_name_2].json
    └── ...

    Args:
        db_path (str): The root directory path of the single database.

    Returns:
        dict: A dictionary with the database name as the key, and its value is a nested 
              dictionary containing all schema, table, and column information.
              Structure: {db_name: {schema_name: {table_name: [column_info, ...]}}}
              Returns an empty dictionary if the path is invalid or the database is empty.
    """
    # 1. Validate if the input path is valid
    if not os.path.isdir(db_path):
        print(f"❌ Error: Database path does not exist or is not a directory -> {db_path}")
        return {}

    # 2. Extract database name from the path
    db_name = os.path.basename(os.path.normpath(db_path))
    print(f"📦 Processing database: {db_name} (Path: {db_path})")

    # 3. Initialize data structure
    db_data = {} # Used to store all schema information
    
    # 4. Iterate through each schema directory
    for schema_name in sorted(os.listdir(db_path)):
        schema_path = os.path.join(db_path, schema_name)
        if not os.path.isdir(schema_path):
            continue
        
        print(f"  📂 Processing schema: {schema_name}")
        schema_data = {} # Used to store all table information for the current schema

        # 5. Iterate through JSON files corresponding to each table (internal logic remains unchanged)
        for file_name in sorted(os.listdir(schema_path)):
            if not file_name.endswith(".json"):
                continue
            
            table_path = os.path.join(schema_path, file_name)
            
            try:
                with open(table_path, "r", encoding="utf-8") as f:
                    table_info = json.load(f)

                # --- Core Logic: Correct table name ---
                original_table_name = table_info.get("table_name")
                table_fullname = table_info.get("table_fullname")
                final_table_name = original_table_name

                if original_table_name and '.' not in original_table_name and table_fullname:
                    parts = table_fullname.split('.')
                    if len(parts) >= 2:
                        corrected_name = '.'.join(parts[-2:])
                        print(f"    - Corrected table name: '{original_table_name}' -> '{corrected_name}'")
                        final_table_name = corrected_name
                
                if not final_table_name:
                    print(f"    - ⚠️ Warning: Valid table name not found in {file_name}, skipped.")
                    continue

                # --- Core Logic: Extract and clean column information ---
                column_names = table_info.get("column_names", [])
                column_types = table_info.get("column_types", [])
                column_descs_raw = table_info.get("description", [None] * len(column_names))
                clean_descs = [desc if desc and str(desc).strip().lower() != "nan" else None for desc in column_descs_raw]
                
                columns_info = []
                for name, type, desc in zip(column_names, column_types, clean_descs):
                    columns_info.append({
                        "column_name": name,
                        "column_type": type,
                        "description": desc
                    })

                schema_data[final_table_name] = columns_info

            except json.JSONDecodeError:
                print(f"    - ⚠️ Warning: File '{file_name}' is not valid JSON, skipped.")
            except Exception as e:
                print(f"    - ❌ Error: Unexpected error processing file '{file_name}': {e}")

        # If there are tables in the schema, add them to the database data
        if schema_data:
            db_data[schema_name] = schema_data
    
    # 6. Assemble the final return result
    if db_data:
        print(f"\n✅ Database '{db_name}' information collection complete!")
        return {db_name: db_data}
    else:
        print(f"\n⚠️ No valid schemas or tables found in database '{db_name}'.")
        return {}

def collect_snowflake_db_info(db_name, credentials):
    """
    Connects to a Snowflake database and collects information about all its schemas, tables, and columns.
    This version ignores any empty schemas that do not contain any tables.
    """
    if isinstance(credentials, str): credentials = json.load(open(credentials, "r", encoding="utf-8"))
    db_structure = {db_name: {}}
    conn = None

    try:
        print(f"🔌 Connecting to Snowflake account '{credentials['account']}', database '{db_name}'...")
        conn = snowflake.connector.connect(
            user=credentials["user"],
            password=credentials["password"],
            account=credentials["account"],
            role=credentials["role"],
            warehouse=credentials["warehouse"],
        )
        print("✅ Connection successful!")
        cursor = conn.cursor()

        print(f"\n🔍 Fetching all schemas in database '{db_name}'...")
        schema_query = f'SHOW SCHEMAS IN DATABASE "{db_name}"'
        
        cursor.execute(schema_query)
        schemas_results = cursor.fetchall()
        schemas_columns = [desc[0] for desc in cursor.description]
        schemas_df = pd.DataFrame(schemas_results, columns=schemas_columns)
        
        print(f"  Found {len(schemas_df)} schemas.")

        for _, schema_row in schemas_df.iterrows():
            schema_name = schema_row['name']
            
            if schema_name == 'INFORMATION_SCHEMA':
                print(f"  - ⏭️  Skipping system schema: {schema_name}")
                continue

            print(f"\n  📂 Processing schema: {schema_name}")
            
            # 【✨✨✨ Core Fix 1/2: Create a temporary dictionary to hold tables for this schema ✨✨✨】
            schema_tables_temp = {}

            table_query = f'SHOW TABLES IN SCHEMA "{db_name}"."{schema_name}"'
            try:
                cursor.execute(table_query)
                tables_results = cursor.fetchall()
                
                if not tables_results:
                    # If no tables, move directly to the next schema loop
                    print(f"    - ℹ️  No tables found in schema '{schema_name}'.")
                    # At the end of the loop, since schema_tables_temp is empty, this schema will be ignored
                else:
                    tables_columns = [desc[0] for desc in cursor.description]
                    tables_df = pd.DataFrame(tables_results, columns=tables_columns)

                    print(f"    - Found {len(tables_df)} tables.")

                    for _, table_row in tables_df.iterrows():
                        raw_table_name = table_row['name']
                        qualified_table_name = f"{schema_name}.{raw_table_name}"
                        
                        print(f"      - Describing table: {qualified_table_name}")
                        desc_query = f'DESCRIBE TABLE "{db_name}"."{schema_name}"."{raw_table_name}"'
                        
                        cursor.execute(desc_query)
                        columns_results = cursor.fetchall()
                        columns_cols = [desc[0] for desc in cursor.description]
                        columns_df = pd.DataFrame(columns_results, columns=columns_cols)
                        
                        columns_info = []
                        for _, col_row in columns_df.iterrows():
                            description = col_row['comment'] if pd.notna(col_row['comment']) else None
                            columns_info.append({
                                "column_name": col_row['name'],
                                "column_type": col_row['type'],
                                "description": description
                            })
                        
                        # Add table info to the temporary dictionary instead of the final structure
                        schema_tables_temp[qualified_table_name] = columns_info
            
            except Exception as table_err:
                print(f"    - ❌ Error fetching tables in schema '{schema_name}': {table_err}")

            # 【✨✨✨ Core Fix 2/2: Add to final structure only if the schema is not empty ✨✨✨】
            if schema_tables_temp:
                db_structure[db_name][schema_name] = schema_tables_temp
                print(f"    - ✅ Schema '{schema_name}' contains data, added.")
            else:
                # This log clearly indicates why we are discarding this schema
                print(f"    - 🗑️  Schema '{schema_name}' is empty and will be ignored.")


    except snowflake.connector.errors.DatabaseError as e:
        print(f"\n❌ Snowflake database error: {e}")
        return {}
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if conn and not conn.is_closed():
            conn.close()
            print("\n🔌 Connection closed.")
            
    return db_structure

def merge_database_info(dict1, dict2):
    """
    Merges two database information dictionaries, taking the union and resolving conflicts based on specific rules.

    Rules:
    1.  The result contains all databases, schemas, and tables from both dictionaries.
    2.  For columns within the same table:
        - If column names match case-insensitively, prioritize the complete column information 
          from dict1 (including name, type, description).
        - If a column exists in only one dictionary, keep it.

    Args:
        dict1 (dict): Priority data source (usually from Snowflake).
        dict2 (dict): Secondary data source (usually from local JSON files).

    Returns:
        dict: The merged database information dictionary.
    """
    print("🚀 Starting database info merge...")
    
    # Step 1: Use a deep copy of dict2 as the base for merging
    merged_dict = copy.deepcopy(dict2)
    print("  - Deep copy of dict2 created as base.")

    # Step 2: Iterate through the priority data source dict1
    for db_name, db_data1 in dict1.items():
        print(f"\n📦 Processing database: {db_name}")
        
        # If the database does not exist in merged_dict, create it
        merged_db = merged_dict.setdefault(db_name, {})
        
        for schema_name, schema_data1 in db_data1.items():
            print(f"  📂 Processing schema: {schema_name}")
            
            # If the schema does not exist in merged_dict, create it
            merged_schema = merged_db.setdefault(schema_name, {})
            
            for table_name, table_cols1 in schema_data1.items():
                print(f"    - Merging table: {table_name}")
                
                table_cols2 = merged_schema.get(table_name, [])

                # Step 3: Core column merging logic
                # Use a dictionary with lowercase column names as keys to handle case-insensitive merging
                final_cols_map = {}
                
                # First, populate the map with columns from dict2
                for col2 in table_cols2:
                    if 'column_name' in col2 and col2['column_name']:
                        final_cols_map[col2['column_name'].lower()] = col2

                # Then, overwrite/add columns from dict1, enforcing dict1 priority
                for col1 in table_cols1:
                    if 'column_name' in col1 and col1['column_name']:
                        final_cols_map[col1['column_name'].lower()] = col1
                        
                # Convert map values back to a list
                merged_cols = list(final_cols_map.values())
                
                # Update the result dictionary with the merged list of columns
                merged_schema[table_name] = merged_cols

    print("\n✅ Database info merge complete!")
    return merged_dict


def compress_database_schema_advanced(full_db_schema, model="deepseek-chat", temperature=1, max_retries=3, retry_delay=2):
    """
    Performs advanced compression on the complete schema of a single database, preserving the Schema hierarchy.

    This function iterates through every Schema in the database, groups and compresses tables within each Schema, 
    and finally returns a dictionary with Schema names as top-level keys.

    Args:
        full_db_schema (dict): The complete database schema, structure: {db_name: {schema_name: {table_data...}}}.
        model (str): Name of the LLM model used for summary generation.
        temperature (float): Temperature parameter for the LLM.
        max_retries (int): Maximum number of retries for LLM calls upon failure.
        retry_delay (int): Seconds to wait before retrying.

    Returns:
        dict: Compressed schema, structure: {schema_name: {compressed_table_data...}}.
    """
    db_name = next(iter(full_db_schema))
    schema_dict = full_db_schema[db_name]
    
    print(f"\n📦 Starting advanced compression for all Schemas in database '{db_name}'...")
    
    llm_params = {"model": model, "temperature": temperature, "max_retries": max_retries, "retry_delay": retry_delay}

    # ==============================================================================
    #  ✨ Internal Helper Functions ✨
    # ==============================================================================
    
    def _find_base_representative_and_extras(table_list, all_tables_data):
        """Finds a base representative table within a group of similar tables and identifies extra columns for each table."""
        base_rep_table = min(table_list, key=lambda t: len(all_tables_data[t]))
        base_columns_set = {col['column_name'].lower() for col in all_tables_data[base_rep_table]}
        extra_columns_map = {}
        for table_name in table_list:
            if table_name == base_rep_table:
                continue
            current_columns = all_tables_data[table_name]
            current_columns_set = {col['column_name'].lower() for col in current_columns}
            extra_col_names = current_columns_set - base_columns_set
            if extra_col_names:
                extra_columns_map[table_name] = [
                    col for col in current_columns if col['column_name'].lower() in extra_col_names
                ]
        return base_rep_table, extra_columns_map

    # ------------------------------------------------------------------------------
    #  ✨✨✨ Using the _generate_table_group_description function provided by you without modification ✨✨✨
    # ------------------------------------------------------------------------------
    def _generate_table_group_description(table_list, base_rep_table, rep_table_structure, extra_cols_info, llm_params):
        """
        Generates a description for a group of similar tables.

        - If the number of tables is <= 5, generates a description using a template, appending context-relevant extra column information.
        - If the number of tables is > 5, calls the LLM with all information (including extra columns) as context to generate a smarter description.
        """
        similar_tables = [t for t in table_list if t != base_rep_table]
        description = ""
        
        if len(table_list) <= 5:
            similar_tables_str = ', '.join(f"'{t}'" for t in similar_tables)
            description = f"Tables {similar_tables_str} and the current table share a similar column pattern."
            
            # --- The extra column grouping logic from the previous version is added and implemented here ---
            if extra_cols_info:
                # Use defaultdict to simplify grouping logic
                cols_to_tables = defaultdict(list)
                for tname, cols in extra_cols_info.items():
                    if cols:
                        extra_col_name = cols[0]['column_name']
                        cols_to_tables[extra_col_name].append(tname)

                group_notes = []
                for col_name, tnames in cols_to_tables.items():
                    quoted_tnames = [f"'{t}'" for t in tnames]
                    
                    if len(quoted_tnames) == 1:
                        tables_str = quoted_tnames[0]
                        group_notes.append(f"table {tables_str} has an extra column like '{col_name}'")
                    else:
                        # Use the "A, B and C" format for a more natural flow
                        tables_str = f"{', '.join(quoted_tnames[:-1])} and {quoted_tnames[-1]}"
                        group_notes.append(f"tables {tables_str} have a common extra column like '{col_name}'")
                
                if group_notes:
                    # Use semicolons to separate different group descriptions for clarity
                    extra_notes = " Additionally: " + "; ".join(group_notes) + "."
                    description += extra_notes
            # --- End of extra column logic ---

        else:
            # For a large number of tables, prepare context and call the LLM
            Table_Name = similar_tables + [base_rep_table]
            column_descriptions = [col['description'] for col in rep_table_structure if col.get('description')]
            Column_Description = str(set(filter(None, column_descriptions)))
            
            # Preprocess extra column information into a string suitable for the prompt
            extra_info_for_prompt = ""
            if extra_cols_info:
                cols_to_tables = defaultdict(list)
                for tname, cols in extra_cols_info.items():
                    if cols:
                        extra_col_name = cols[0]['column_name']
                        cols_to_tables[extra_col_name].append(tname)

                group_notes = []
                for col_name, tnames in cols_to_tables.items():
                    quoted_tnames = [f"'{t}'" for t in tnames]
                    if len(quoted_tnames) == 1:
                        tables_str = quoted_tnames[0]
                        group_notes.append(f"table {tables_str} has an extra column named '{col_name}'")
                    else:
                        tables_str = f"{', '.join(quoted_tnames[:-1])} and {quoted_tnames[-1]}"
                        group_notes.append(f"tables {tables_str} share a common extra column named '{col_name}'")
                
                if group_notes:
                    extra_info_for_prompt = "\n\nImportant additional context: " + "; ".join(group_notes) + "."
            
            # Call the LLM with the complete context
            attempt = 0
            while attempt < llm_params['max_retries']:
                try:
                    # Assuming prompt, LLM_output, and extract_and_parse_json are defined
                    formatted_prompt = prompt.format(
                        Table_Name=Table_Name, 
                        Column_Description=Column_Description,
                        Extra_info=extra_info_for_prompt
                    )
                    messages = [{"role": "user", "content": formatted_prompt}]
                    print("--- Calling LLM with enriched prompt ---")
                    print("prompt message:", messages)
                    _, _, Thinking, LLM_return = LLM_output(messages=messages, model=llm_params['model'], temperature=llm_params['temperature'])
                    print("LLM Thinking:", Thinking)
                    print("LLM return:", LLM_return)
                    temp = extract_and_parse_json(LLM_return)
                    description = next(iter(temp.values()))
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= llm_params['max_retries']:
                        similar_tables_str = ', '.join(f"'{t}'" for t in similar_tables)
                        description = f"Tables {similar_tables_str} and the current table share a similar column pattern."
                        print("❌ Reached maximum retry attempts, using default description.")
                        break
                    print(f"⚠️ LLM call failed (attempt {attempt}), retrying... Error: {e}")
                    time.sleep(llm_params['retry_delay'])
                
        return description
    # ------------------------------------------------------------------------------
    #  ✨✨✨ Function End ✨✨✨
    # ------------------------------------------------------------------------------

    def _format_columns_to_list(columns_as_dicts):
        """Converts a list of column information dictionaries into a list of the target format."""
        return [[c.get('column_name'), c.get('column_type'), c.get('description')] for c in columns_as_dicts]
    
    # ------------------------------------------------------------------------------
    #  ✨✨✨ Newly Added Internal Function ✨✨✨
    # ------------------------------------------------------------------------------
    def _collect_and_save_all_columns_as_paths(full_db_schema_input, db_name_input):
        """
        Collects full paths of all columns in the database and saves them to a pickle file.
        
        Path format: db_name.schema_name.table_name.column_name
        Directly extracted from the raw `full_db_schema` to ensure data integrity.
        """
        print(f"\n🔍 Starting collection of all column paths...")
        all_column_paths = set()
        
        # Variable Analysis:
        # - db_name_input: Database name passed from outside the function, e.g., 'ORACLE_SQL'
        # - full_db_schema_input: Complete original schema, structure {db_name: {schema_name: {table_name: [...]}}}
        
        # Iterate through Schemas
        # `full_db_schema_input[db_name_input]` -> {schema_name: {table_name: [...]}}
        for schema_name, tables_dict in full_db_schema_input[db_name_input].items():
            # Iterate through tables within the Schema
            # `tables_dict` -> {table_name: [{'column_name': ...}, ...]}
            for table_name, columns_list in tables_dict.items():
                # Iterate through columns in the table
                # `columns_list` -> [{'column_name': 'col1', ...}, {'column_name': 'col2', ...}]
                for column_dict in columns_list:
                    column_name = column_dict.get('column_name')
                    if column_name:
                        # Construct full path string
                        # Format: 'ORACLE_SQL.ORACLE_SQL.MONTHLY_SALES.mth'
                        full_path = f"{db_name_input}.{table_name}.{column_name}"
                        all_column_paths.add(full_path)

        # Save to pickle file
        print("all_column_paths:",all_column_paths)
        db_dir = os.path.join(snow_DB_dir, db_name_input)
        filename=os.path.join(db_dir, f"{db_name_input}_all_col.pkl")
        try:
            with open(filename, 'wb') as f:
                pickle.dump(all_column_paths, f)
            
            print(f"  - ✅ Success! A total of {len(all_column_paths)} unique column paths saved to '{filename}'.")
        except Exception as e:
            print(f"  - ❌ Error: Could not save column paths to file '{filename}'. Reason: {e}")
            
        return all_column_paths # Optionally return the set
    # ------------------------------------------------------------------------------
    #  ✨✨✨ New Function End ✨✨✨
    # ------------------------------------------------------------------------------


    # ==============================================================================
    #  ✨ Main Process: Iterate through every Schema (This logic remains unchanged) ✨
    # ==============================================================================
    
    final_database_output = {}

    for schema_name, all_tables_in_this_schema in schema_dict.items():
        print(f"\n--- Processing Schema: '{schema_name}' ---")
        
        if not all_tables_in_this_schema:
            print(f"  - Schema '{schema_name}' is empty, skipping.")
            continue

        compressed_tables = {}
        table_info = {}
        desc_summary = {}
        
        table_groups = defaultdict(list)
        
        for tname in all_tables_in_this_schema.keys():
            base_name = remove_digits(tname)
            table_groups[base_name].append(tname)
        
        for base_name, table_list in table_groups.items():
            if len(table_list) == 1:
                table_name = table_list[0]
                compressed_tables[table_name] = all_tables_in_this_schema[table_name]
                continue
            
            table_list.sort()
            base_rep_table, extras_map = _find_base_representative_and_extras(table_list, all_tables_in_this_schema)
            
            compressed_tables[base_rep_table] = all_tables_in_this_schema[base_rep_table]
            table_info[base_rep_table] = [t for t in table_list if t != base_rep_table]
            
            description = _generate_table_group_description(
                table_list, 
                base_rep_table, 
                all_tables_in_this_schema[base_rep_table], 
                extras_map, 
                llm_params
            )
            desc_summary[base_rep_table] = description

        current_schema_output = {}
        
        current_schema_output.update({tname: _format_columns_to_list(cols) for tname, cols in compressed_tables.items()})
        
        if table_info:
            current_schema_output["table_Information"] = table_info
        
        if desc_summary:
            current_schema_output["table_description_summary"] = desc_summary
            
        final_database_output[schema_name] = current_schema_output
        print(f"  - ✅ Schema '{schema_name}' compression finished.")

    print(f"\n✅ All schemas in database '{db_name}' have been compressed!")
    
    # ==============================================================================
    # ✨ Before returning, call the new function to collect and save all column paths ✨
    # ==============================================================================
    _collect_and_save_all_columns_as_paths(full_db_schema, db_name)
    
    return final_database_output


def remove_empty_columns_from_schema(compressed_schema, db_name, credentials):
    """
    Removes completely empty columns from the compressed schema organized by Schema, based on actual data.

    This function iterates through each Schema, checks and prunes empty columns for tables within it, 
    while ensuring metadata like `table_Information` and `table_description_summary` are preserved intact.

    Args:
        compressed_schema (dict): The dictionary generated by compress_database_schema_advanced, 
                                  keyed by Schema name.
        db_name (str): The name of the database, used to construct queries.
        credentials (dict): Credentials required to connect to Snowflake.

    Returns:
        dict: A new dictionary where completely empty columns have been removed, maintaining the original Schema structure and metadata.
    """
    print(f"\n🔬 Starting empty column check based on real data from database '{db_name}' (Schema by Schema)...")
    if isinstance(credentials, str):
        credentials = json.load(open(credentials, "r", encoding="utf-8"))
    
    # Create a deep copy for operation to ensure original data is unaffected
    pruned_schema = copy.deepcopy(compressed_schema)
    
    # ==============================================================================
    #  ✨ Internal Helper Function (_get_column_count) - Logic correct, no changes needed ✨
    # ==============================================================================
    
    def _get_column_count(fully_qualified_table_name, column_name):
        """
        Executes SQL query to get the number of non-null rows for a column.
        - Defaults to executing with double quotes.
        - If invalid identifier error occurs, falls back to querying without quotes.
        - Table or column does not exist/no permission -> Returns 0 (treated as empty column).
        - Other unknown errors -> Returns -1 (treated as non-empty).
        """
        conn = None

        parts = fully_qualified_table_name.split(".")
        if len(parts) == 3:
            database_name, schema_name, table_name_raw = parts
        elif len(parts) == 2:
            database_name = db_name
            schema_name, table_name_raw = parts
        else:
            print(f"⚠️ Table name '{fully_qualified_table_name}' format error, cannot parse.")
            return -1

        def run_query(cursor, col_expr):
            query = f'SELECT COUNT({col_expr}) FROM "{database_name}"."{schema_name}"."{table_name_raw}"'
            print(f"    - Executing query: {query}")
            cursor.execute(query)
            result = cursor.fetchone()
            return result[0] if result else 0

        try:
            conn = snowflake.connector.connect(
                user=credentials["user"],
                password=credentials["password"],
                account=credentials["account"],
                role=credentials["role"],
                warehouse=credentials["warehouse"],
            )
            cursor = conn.cursor()

            try:
                return run_query(cursor, f'"{column_name}"')

            except ProgrammingError as e:
                err_msg = str(e).lower()
                if "does not exist" in err_msg or "not authorized" in err_msg:
                    print(f"    - ⚠️ Table does not exist/No permission: {fully_qualified_table_name} -> Treated as empty column")
                    return 0
                if "invalid identifier" in err_msg:
                    print(f"    - ⚠️ Column name case might not match, retrying without quotes")
                    try:
                        return run_query(cursor, column_name)
                    except ProgrammingError as e2:
                        print(f"    - ⚠️ Second query failed: {e2} -> Column does not exist -> Treated as empty column")
                        return 0  # ✅ Second attempt also failed -> Treat as column does not exist
                print(f"    - ⚠️ Query failed: {e} -> Treated as non-empty")
                return -1
        except Exception as e:
            print(f"❌ Unexpected error during connection or execution: {e} -> Treated as non-empty")
            return -1
        finally:
            if conn and not conn.is_closed():
                conn.close()


    # ==============================================================================
    #  ✨ Main Process Refactoring: Iterate through every Schema ✨
    # ==============================================================================

    # Introduce outer loop to process each Schema individually
    for schema_name, schema_content in pruned_schema.items():
        print(f"\n🔍 Analyzing Schema: '{schema_name}'")
        
        # Get metadata from the content of the current Schema
        table_info = schema_content.get("table_Information", {})
        
        # Find all representative tables under the current Schema (values are lists of columns)
        representative_tables = [
            key for key, value in schema_content.items()
            if isinstance(value, list)
        ]

        # Perform column check for each representative table in the current Schema
        for rep_table in representative_tables:
            print(f"  - Checking representative table '{rep_table}' and its similar tables...")
            
            tables_to_check = [rep_table] + table_info.get(rep_table, [])
            # Get original column information from the content of the current Schema
            original_columns = schema_content[rep_table]
            columns_to_keep = []
            
            for column_data in original_columns:
                column_name = column_data[0]
                is_empty_everywhere = True
                
                print(f"    - Checking column '{column_name}'...")
                
                for table_to_check in tables_to_check:
                    count = _get_column_count(table_to_check, column_name)
                    
                    # If non-empty data is found in any table (count != 0), keep the column and stop checking
                    if count != 0:
                        print(f"      ❗ Found non-empty data in table '{table_to_check}' (count={count}). This column will be kept.")
                        is_empty_everywhere = False
                        break 
                
                if not is_empty_everywhere:
                    columns_to_keep.append(column_data)
                else:
                    print(f"      ✅ Column '{column_name}' is empty in all {len(tables_to_check)} related tables. It will be removed.")
            
            # ✨ Key: Update the column list of the corresponding table in the current Schema content
            # Since schema_content is a reference to the dictionary within pruned_schema, modifications here will be directly reflected in the final result
            schema_content[rep_table] = columns_to_keep

    print("\n✅ Empty column check and pruning complete!")
    return pruned_schema

def format_value(value, max_len=50):

    if value is None:
        return None
    if isinstance(value, bytearray):
        processed_value = value.decode("utf-8", errors="ignore")
    else:
        processed_value = value

    if isinstance(processed_value, str):
        if len(processed_value) > max_len:
            return processed_value[:max_len-3] + '...'
        return processed_value
    
    return processed_value

def format_examples_string(example_vals: list) -> str:
    """
    Formats a list of values into an example string without any quotes, using direct string conversion and concatenation.
    E.g., ['a', 123, '2022-01-01'] -> 'examples: [a, 123, 2022-01-01]'
    """
    if not example_vals:
        return "examples: []"

    # Core logic: Call str() on each value, then join them directly with commas.
    # This ensures the final result absolutely contains no quotes.
    string_parts = [str(val) for val in example_vals]
    
    inner_content = ", ".join(string_parts)
    
    return f"examples: [{inner_content}]"


# --- Main Function ---
def enrich_schema_with_examples(db_schema: dict, db_name: str, credentials: dict, example_limit: int = 3):
    """
    Enriches the schema information dictionary with example data queried from the Snowflake database.
    """
    conn = None
    try:
        conn = snowflake.connector.connect(
            user=credentials["user"],
            password=credentials["password"],
            account=credentials["account"],
            role=credentials["role"],
            warehouse=credentials["warehouse"],
        )
        cursor = conn.cursor()
        print(f"Successfully connected to Snowflake database: '{db_name}'.")
    except Exception as e:
        print(f"FATAL: Failed to connect to Snowflake. Error: {e}")
        return db_schema

    try:
        for schema_name, tables in db_schema.items():
            if not isinstance(tables, dict): continue
            print(f"\nProcessing Schema: '{schema_name}'")

            for fq_table_name, content in tables.items():
                if not isinstance(content, list) or not content: continue
                columns_info_list = content
                print(f"  Processing Table: '{fq_table_name}'")

                try:
                    parsed_schema, simple_table_name = fq_table_name.split('.', 1)
                    full_table_path = f'"{db_name}"."{parsed_schema}"."{simple_table_name}"'
                except ValueError:
                    print(f"  WARNING: Cannot parse schema/table from '{fq_table_name}'. Skipping.")
                    continue

                all_col_names = [info[0] for info in columns_info_list if isinstance(info, list) and len(info) > 0]
                if not all_col_names: continue
                col_samples = defaultdict(list)

                # --- 2a. Efficient Bulk Sampling Query ---
                try:
                    quoted_cols = ', '.join([f'"{col}"' for col in all_col_names])
                    query_bulk = f'SELECT {quoted_cols} FROM {full_table_path} SAMPLE ROW (10000 ROWS)'
                    print(f"    Executing bulk query (fetching 1000 rows)...")
                    cursor.execute(query_bulk)
                    sample_col_names = [desc[0] for desc in cursor.description]
                    sample_rows = cursor.fetchall()

                    df = pd.DataFrame(sample_rows, columns=sample_col_names)
                    print(f"    Bulk fetched shape: {df.shape}")

                    if not df.empty:
                        df = df.dropna(how='any')
                        print(f"    After dropping rows with any NaN: {df.shape}")

                        for original_col_name in all_col_names:
                            if original_col_name.upper() in df.columns:
                                series = df[original_col_name.upper()]
                                series_formatted = series.apply(format_value)
                                non_null_vals = series_formatted.unique().tolist()[:example_limit]
                                col_samples[original_col_name.upper()] = non_null_vals

                except ProgrammingError as e:
                    print(f"    INFO: Bulk query for '{fq_table_name}' failed. Error: {e}. Will use fallback.")

                # --- 2b. Fallback Mechanism ---
                for col_info in columns_info_list:
                    col_name = col_info[0]
                    cname_upper = col_name.upper()

                    if len(col_samples.get(cname_upper, [])) < example_limit:
                        def try_fallback(quote: bool):
                            needed = example_limit - len(col_samples[cname_upper])
                            if needed <= 0: return

                            col_expr = f'"{col_name}"' if quote else col_name
                            query_fallback = (
                                f'SELECT DISTINCT {col_expr} '
                                f'FROM {full_table_path} SAMPLE ROW (10000 ROWS) '
                                f'WHERE {col_expr} IS NOT NULL '
                                f'LIMIT {needed}'
                            )
                            print(f"      Fallback for column '{col_name}' (quoted={quote})...")
                            cursor.execute(query_fallback)
                            
                            for row in cursor.fetchall():
                                val = format_value(row[0])
                                if val is not None and val not in col_samples[cname_upper]:
                                    col_samples[cname_upper].append(val)

                        try: try_fallback(quote=True)
                        except ProgrammingError:
                            try: try_fallback(quote=False)
                            except ProgrammingError as e2:
                                print(f"      WARNING: Unquoted fallback also failed for '{col_name}'. Error: {e2}")
                
                # --- 2c. Update Column Information ---
                for col_info in columns_info_list:
                    cname_upper = col_info[0].upper()
                    example_vals = col_samples.get(cname_upper, [])
                    
                    # 🔹 Call the new helper function to generate a clean string
                    examples_str = format_examples_string(example_vals)
                    
                    col_info.append(examples_str)

    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if conn and not conn.is_closed():
            conn.close()
            print("\nConnection to Snowflake closed.")

    return db_schema


def process_and_save_database(db_name, db_root, model,credentials=SNOWFLAKE_CREDENTIALS, overwrite=True):
    """
    Executes the complete workflow from data collection, merging, compression, and pruning to final saving.
    """
    # --- Step 0: Check if the compressed file already exists ---
    output_dir = os.path.join(db_root, db_name)
    output_path = os.path.join(output_dir, f"{db_name}_M-Schema.json")
    
    if os.path.exists(output_path) and not overwrite:
        print(f"⚠️ Detected existing compressed file: {output_path}, and overwrite=False. Skipping processing.")
        return
    if overwrite and os.path.exists(output_path):
        print(f"♻️ Overwrite mode enabled. Reprocessing database '{db_name}'.")

    try:
        # --- Step 1: Data Collection ---
        print(f"\n--- Step 1: Collect Data ---")
        dist2 = collect_snowflake_db_info(db_name=db_name, credentials=credentials)
        dist1 = collect_single_database_info(f"{db_root}/{db_name}")

        # --- Step 2: Merging ---
        print(f"\n--- Step 2: Merge Data (JSON Priority) ---")
        merged_dist = merge_database_info(dist1, dist2)

        # --- Step 3: Compression ---
        print(f"\n--- Step 3: Compress Schema ---")
        dist3 = compress_database_schema_advanced(full_db_schema=merged_dist,model=model)

        # --- Step 4: Prune Empty Columns ---
        print(f"\n--- Step 4: Prune Empty Columns ---")
        dist4 = remove_empty_columns_from_schema(
            compressed_schema=dist3,
            db_name=db_name,
            credentials=credentials
        )

        # --- Step 5: Example Retrieval ---
        print(f"\n--- Step 5: Example Retrieval ---")
        dist5 = enrich_schema_with_examples(
            db_schema=dist4,  # It's best to use the pruned version here
            db_name=db_name,
            credentials=credentials
        )

        # --- Step 6: Saving ---
        print(f"\n--- Step 6: Save Final Result ---")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dist5, f, indent=4, ensure_ascii=False)

        print(f"✅ File saved successfully: {output_path}")

    except Exception as e:
        # Do not swallow the exception here; raise it for the outer layer to catch
        raise RuntimeError(f"Failed to process database '{db_name}': {e}") from e






if __name__ == "__main__":
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(description="Spider2 Snow Preprocessing")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--log_path", type=str, default="preprocessing_Snowfalke.log", help="Log file path")
    parser.add_argument("--status_path", type=str, default="Snowfalke_statu.json", help="Status file path")
    
    args = parser.parse_args()

    # --- Variable Assignment ---
    databases_root = snow_DB_dir
    model = args.model
    # Ensure default paths are absolute (relative to the current execution directory)
    log_path = os.path.abspath(args.log_path) 
    status_json_path = os.path.abspath(args.status_path)

    # Log Settings
    log_file = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    # Print start time and status messages
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n\n\n------------------------------------datetime：{current_time}------------------------------------\n\n\n")
    print(f"The data preprocessing work for the **Spider2-snow** task is currently in progress!")
    print(f"Model: {model} | Workers: {args.workers} | Input: {databases_root}")

    success_list = []
    fail_list = []
    
    def process_one_db(db_name):
        """Task function for parallel execution"""
        try:
            # Note: Ensure process_and_save_database is available in the current context
            process_and_save_database(db_name, databases_root, model)
            return ("success", db_name)
        except Exception as e:
            print(f"❌❌❌ Failed to process database {db_name}. Error info: {e}")
            return ("fail", db_name)

    # Validation check: Ensure input directory exists
    if not os.path.exists(databases_root):
        print(f"Error: Input path does not exist -> {databases_root}")
        sys.exit(1)

    # Get all database directories
    db_names = [d for d in sorted(os.listdir(databases_root)) 
           if os.path.isdir(os.path.join(databases_root, d)) 
           and d != "AMAZON_VENDOR_ANALYTICS__SAMPLE_DATASET"]

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_db = {executor.submit(process_one_db, db_name): db_name for db_name in db_names}

        for future in as_completed(future_to_db):
            status, db_name = future.result()
            if status == "success":
                success_list.append(db_name)
            else:
                fail_list.append(db_name)

    # Save status record to JSON
    final_status = {
        "success": success_list,
        "fail": fail_list
    }
    with open(status_json_path, "w", encoding="utf-8") as f:
        json.dump(final_status, f, indent=4, ensure_ascii=False)
        print(f"\n✅ All processing complete. Status info saved to: {status_json_path}")

    # Restore stdout & close log file
    sys.stdout = original_stdout
    log_file.close()



