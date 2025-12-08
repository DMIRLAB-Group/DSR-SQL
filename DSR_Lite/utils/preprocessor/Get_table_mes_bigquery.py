#Given that queries to this database will incur costs, we will not make actual calls here and will only use the official sample documents provided, which may impact performance.

import os
import json
import re
from pathlib import Path
import argparse
from collections import defaultdict
import time
import sys


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from LLM.LLM_OUT import LLM_output
from utils.extract_json import extract_and_parse_json
from utils.DBsetup.Get_DB import read_db_config


# --- Basic I/O Functions ---
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return None

def write_json_file(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ Successfully generated schema file at: {file_path}")
    except Exception as e:
        print(f"❌ Error writing to {file_path}: {e}")

# --- Example Formatting Functions ---
def format_value(value, max_len=50):
    if value is None:
        return None
    if isinstance(value, bytearray):
        processed_value = value.decode("utf-8", errors="ignore")
    else:
        processed_value = value
    if isinstance(processed_value, str) and len(processed_value) > max_len:
        return processed_value[:max_len-3] + '...'
    return processed_value

def format_examples_string(example_vals: list) -> str:
    if not example_vals:
        return ""
    string_parts = [str(val) for val in example_vals]
    return f"examples: [{', '.join(string_parts)}]"

def get_formatted_examples_for_column(column_name, sample_rows):
    if not sample_rows:
        return format_examples_string([])
    unique_examples = set()
    formatted_examples_list = []
    for row in sample_rows:
        if len(unique_examples) >= 3:
            break
        if isinstance(row, dict) and column_name in row:
            original_value = row.get(column_name)
            if original_value is not None:
                if str(original_value) not in unique_examples:
                    unique_examples.add(str(original_value))
                    formatted_examples_list.append(format_value(original_value))
    return format_examples_string(formatted_examples_list)

# --- Series Table Identification ---
def remove_digits(s):
    if not isinstance(s, str): s = str(s)
    return re.sub(r'\d', '', s)

def group_tables_by_series(table_json_paths):
    groups = defaultdict(list)
    for path in table_json_paths:
        groups[remove_digits(path.stem)].append(path)
    return groups

# --- Core Processing Logic ---
def format_table_schema(table_data, aggregated_sample_rows=None):
    # Initialize the list to store formatted schema information
    schema_list = []
    
    # Extract basic table metadata
    descriptions = table_data.get('description', [])
    column_names = table_data.get('column_names', [])
    column_types = table_data.get('column_types', [])
    
    # Use aggregated samples if provided, otherwise use the table's own samples
    sample_rows_to_use = aggregated_sample_rows if aggregated_sample_rows is not None else table_data.get('sample_rows', [])
    
    # Ensure the descriptions list is as long as the column names list
    if len(descriptions) < len(column_names):
        descriptions.extend([None] * (len(column_names) - len(descriptions)))
        
    # Iterate over columns to build the schema definition
    for i, col_name in enumerate(column_names):
        col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
        col_desc = descriptions[i] if i < len(descriptions) else None
        examples_str = get_formatted_examples_for_column(col_name, sample_rows_to_use)
        schema_list.append([col_name, col_type, col_desc, examples_str])
    return schema_list

def process_single_table(table_path, dataset_id):
    # Read the individual table file
    table_data = read_json_file(table_path)
    if not table_data: return {}
    
    # Construct a unique ID for the table
    formatted_table_id = f"{dataset_id}.{table_data['table_name']}"
    
    # Generate the formatted schema
    schema = format_table_schema(table_data)
    return {formatted_table_id: schema}

def process_table_series(table_paths, dataset_id, model):
    # Sort paths to determine the representative table (usually the first one)
    sorted_paths = sorted(table_paths)
    rep_path = sorted_paths[0]
    
    # Load data for all tables in the series
    all_tables_data = {p.stem: read_json_file(p) for p in sorted_paths}
    rep_data = all_tables_data.get(rep_path.stem)
    if not rep_data: return {}
    rep_table_id = f"{dataset_id}.{rep_data['table_name']}"
    
    # Aggregate sample rows from all tables to provide better examples
    aggregated_sample_rows = [row for td in all_tables_data.values() if td and 'sample_rows' in td for row in td['sample_rows']]
    
    # Identify columns that exist in other tables but are missing in the representative table
    rep_cols = set(rep_data.get('column_names', []))
    extra_cols_info = defaultdict(list)
    all_table_names_in_series = [f"{dataset_id}.{td['table_name']}" for td in all_tables_data.values() if td]
    
    for path in sorted_paths:
        td = all_tables_data.get(path.stem)
        if not td: continue
        current_cols = set(td.get('column_names', []))
        extra_cols = current_cols - rep_cols
        if extra_cols:
            full_table_id = f"{dataset_id}.{td['table_name']}"
            extra_cols_info[full_table_id].append({"column_name": list(extra_cols)[0]})
    
    # Configure LLM parameters and context
    llm_params = {'max_retries': 5, 'retry_delay': 1, 'model': model, 'temperature': 0}
    rep_table_structure_for_prompt = [{'column_name': name, 'description': desc} for name, desc in zip(rep_data.get('column_names', []), rep_data.get('description', []))]
    
    # Generate a summarized description for the entire table group
    description = _generate_table_group_description(all_table_names_in_series, rep_table_id, rep_table_structure_for_prompt, extra_cols_info, llm_params)
    
    # Construct the final output object linking the series to the representative table
    output = {
        rep_table_id: format_table_schema(rep_data, aggregated_sample_rows=aggregated_sample_rows),
        "table_Information": {rep_table_id: [name for name in all_table_names_in_series if name != rep_table_id]},
        "table_description_summary": {rep_table_id: description}
    }
    return output


def _generate_table_group_description(table_list, base_rep_table, rep_table_structure, extra_cols_info, llm_params):
    # (此函数保持不变，直接使用您提供的版本)
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
...
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
{extra_cols_info}

## Output format(Markdown)
<Analysis Summary>Analysis Summary put thhere</Analysis Summary>
**return**
```json
{{
"Answer":"Please provide the answer here (refer to the example)!",
}}
```
'''

    similar_tables = [t for t in table_list if t != base_rep_table]
    description = ""
    
    if len(table_list) <= 5:
        similar_tables_str = ', '.join(f"'{t}'" for t in similar_tables)
        description = f"Tables {similar_tables_str} and the current table '{base_rep_table}' share a similar column pattern."
        
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
                    group_notes.append(f"table {tables_str} has an extra column '{col_name}'")
                else:
                    tables_str = f"{', '.join(quoted_tnames[:-1])} and {quoted_tnames[-1]}"
                    group_notes.append(f"tables {tables_str} have a common extra column '{col_name}'")
            
            if group_notes:
                extra_notes = " Additionally: " + "; ".join(group_notes) + "."
                description += extra_notes
    else:
        Table_Name_dict = {base_rep_table: similar_tables} # 格式化为字典
        column_descriptions = [col['description'] for col in rep_table_structure if col.get('description')]
        Column_Description = str(set(filter(None, column_descriptions)))
        
        extra_info_for_prompt = ""
        if extra_cols_info:
            extra_info_for_prompt = json.dumps(dict(extra_cols_info), indent=2)

        attempt = 0
        while attempt < llm_params['max_retries']:
            try:
                formatted_prompt = prompt.format(
                    Table_Name=json.dumps(Table_Name_dict, indent=2), 
                    Column_Description=Column_Description,
                    extra_cols_info=extra_info_for_prompt
                )
                messages = [{"role": "user", "content": formatted_prompt}]
                print(f"--- Calling LLM for table group starting with '{base_rep_table}' ---")
                _, _, Thinking, LLM_return = LLM_output(messages=messages, model=llm_params['model'], temperature=llm_params['temperature'])
                
                temp = extract_and_parse_json(LLM_return)
                if not temp or "Answer" not in temp:
                    raise ValueError("LLM returned empty or invalid JSON.")
                description = temp["Answer"]
                print("--- LLM call successful ---")
                break
            except Exception as e:
                attempt += 1
                print(f"⚠️ LLM call failed (attempt {attempt}), retrying... Error: {e}")
                if attempt >= llm_params['max_retries']:
                    similar_tables_str = ', '.join(f"'{t}'" for t in similar_tables)
                    description = f"Tables {similar_tables_str} and the current table '{base_rep_table}' share a similar column pattern."
                    print("❌ Reached maximum retry attempts, using default description.")
                    break
                time.sleep(llm_params['retry_delay'])
    return description

# --- Main Process Control (Updated) ---
def process_dataset_folder(dataset_path, model):
    """Process a single project_id.dataset_id folder and return project_id, dataset_id, and table data."""
    dataset_folder_name = dataset_path.name
    try:
        project_id, dataset_id = dataset_folder_name.split('.', 1)
    except ValueError:
        print(f"Skipping folder with unexpected name format: {dataset_folder_name}")
        return None, None, {}

    print(f"\nProcessing Project: '{project_id}', Dataset: '{dataset_id}'")
    table_json_paths = list(dataset_path.glob('*.json'))
    if not table_json_paths:
        print(f"No JSON files found in {dataset_path}")
        return project_id, dataset_id, {}

    table_groups = group_tables_by_series(table_json_paths)

    # 1. Create independent accumulation dictionaries
    all_table_schemas = {}
    aggregated_table_information = {}
    aggregated_table_summaries = {}

    for base_name, paths in table_groups.items():
        if len(paths) == 1:
            # Process single tables normally and update the schema dictionary
            single_table_schema = process_single_table(paths[0], dataset_id)
            all_table_schemas.update(single_table_schema)
        else:
            # Process table series
            series_result = process_table_series(paths, dataset_id, model)
            
            # 2. Separate different parts from the results
            # Use .pop() to safely extract/remove special keys, leaving only the table schema
            series_info = series_result.pop("table_Information", {})
            series_summary = series_result.pop("table_description_summary", {})
            
            # The remainder is the schema, update it into the main dictionary
            all_table_schemas.update(series_result)
            
            # 3. Update separated info into corresponding accumulation dictionaries
            # .update() merges internal dictionaries of info and summary correctly
            aggregated_table_information.update(series_info)
            aggregated_table_summaries.update(series_summary)

    # 4. After the loop, combine all dictionaries into the final output
    final_dataset_output = {}
    final_dataset_output.update(all_table_schemas) # Add all table schemas first

    # Only add these keys if table series exist
    if aggregated_table_information:
        final_dataset_output["table_Information"] = aggregated_table_information
    if aggregated_table_summaries:
        final_dataset_output["table_description_summary"] = aggregated_table_summaries

    return project_id, dataset_id, final_dataset_output


def main(task_folder_path, model):
    """Main function to process all databases under the specified task folder and build the correct project -> dataset hierarchy."""
    task_path = Path(task_folder_path)
    if not task_path.is_dir():
        print(f"Error: Provided path '{task_folder_path}' is not a valid directory.")
        return

    task_name = task_path.name
    print(f"--- Starting processing for task: {task_name} ---")

    final_output = {}
    dataset_folders = [d for d in task_path.iterdir() if d.is_dir()]
    
    for dataset_path in dataset_folders:
        project_id, dataset_id, tables_data = process_dataset_folder(dataset_path, model)
        if not project_id or not dataset_id:
            continue
        if project_id not in final_output:
            final_output[project_id] = {}
        final_output[project_id][dataset_id] = tables_data

    output_file_path = task_path / f"{task_name}_M-Schema.json"
    write_json_file(final_output, output_file_path)
    print(f"--- Finished processing for task: {task_name} ---")


if __name__ == '__main__':
    # ==================== Core Modification Start ====================
    # 1. Initialize command-line argument parser
    parser = argparse.ArgumentParser() # <-- Added
    # 2. Add --model argument and set default value
    parser.add_argument("--model", default="deepseek-chat", help="Specify the model to use.") # <-- Added
    # 3. Parse arguments
    args = parser.parse_args() # <-- Added
    model = args.model # <-- Modified: Get model from command-line arguments
    # ==================== Core Modification End ====================

    # The code below remains completely unchanged
    _,_,BASE_DIRECTORY,_,_ = read_db_config()
    
    # 2. Create path object using pathlib
    base_path = Path(BASE_DIRECTORY)

    # 3. Check if base directory exists, exit with error if not
    if not base_path.is_dir():
        print(f"❌ Error: Base directory not found at '{BASE_DIRECTORY}'")
        sys.exit(1) # Can use sys module to exit the program after importing it
    
    # 4. Get list of all subfolders
    # Using sorted() ensures consistent execution order for easier tracking
    task_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not task_folders:
        print(f"🤷 No subdirectories found in '{BASE_DIRECTORY}' to process.")
    else:
        print(f"🚀 Starting batch processing for {len(task_folders)} tasks in: {base_path}")
        print(f"🏷️ Using model: {model}") # Print current model in use
        print("==========================================================")
        
        # 5. Iterate through all subfolders and call main function for each
        for i, folder_path in enumerate(task_folders):
            print(f"\n[{i + 1}/{len(task_folders)}] ==> Processing directory: {folder_path.name}")
            # main function accepts Path object as argument
            main(folder_path, model)
            print(f"==> Finished processing directory: {folder_path.name}")
            print("----------------------------------------------------------")

        print("\n==========================================================")
        print("✅ Batch processing complete. All directories have been processed.")
