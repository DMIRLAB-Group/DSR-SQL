import os
import sys
import json
import argparse
from datetime import datetime

# Get the absolute path of the current script, then go up one level to the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from LLM.LLM_OUT import LLM_output
from utils.extract_json import extract_and_parse_json
from utils.Prompt import *
from utils.Database_Interface import detect_db_type

# Global variable for the log path, will be set from command-line arguments.
LOG_PATH = None

def log_msg(msg):
    """Logs a message with a timestamp to both a file and the console."""
    if LOG_PATH is None:
        print("Error: LOG_PATH is not set. Please configure it via command-line arguments.")
        return
        
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().isoformat()
    full_msg = f"[{timestamp}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(full_msg + "\n")
    print(full_msg)

def read_md_file(base_path, md_filename):
    """Reads the content of a Markdown file."""
    md_path = os.path.join(base_path, md_filename)
    log_msg(f"Attempting to read Markdown file: {md_path}")
    
    if not os.path.exists(md_path):
        log_msg(f"[ERROR] File not found: {md_path}")
        return ""
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log_msg(f"Successfully read Markdown file: {md_filename}, length={len(content)}")
        return content
    except Exception as e:
        log_msg(f"[ERROR] Failed to read {md_path}: {e}")
        return ""

def extract_evidence(EK, question, max_retries=3,model="deepseek-chat"):
    """
    Constructs a prompt, sends it to the LLM, and extracts the 'evidence' field from the response.
    Logs the full LLM input and output.
    """
    KC = Knowledge_Compression(Question=question, Knowledge=EK)
    KC_mess = [{"role": "user", "content": KC.Prompt}]
    
    log_msg(f"Constructed LLM prompt. Length={len(KC.Prompt)}")
    log_msg("--- LLM INPUT ---")
    log_msg(json.dumps(KC_mess, indent=2, ensure_ascii=False))
    log_msg("--- END LLM INPUT ---")

    for attempt in range(1, max_retries + 1):
        try:
            _,_,Thinking, LLM_return = LLM_output(
                messages=KC_mess,
                model=model,
                temperature=KC.temperature
            )

            log_msg(f"[Attempt {attempt}] Received {model} LLM output.")
            log_msg("--- LLM THINKING ---")
            log_msg(Thinking)
            log_msg("--- END LLM THINKING ---")
            log_msg("--- LLM RETURN ---")
            log_msg(LLM_return)
            log_msg("--- END LLM RETURN ---")

            ge_evidence = extract_and_parse_json(text=LLM_return)
            if "evidence" in ge_evidence:
                log_msg(f"[Attempt {attempt}] Successfully extracted 'evidence' field.")
                return ge_evidence["evidence"]
            else:
                log_msg(f"[Attempt {attempt}] 'evidence' field is missing in the JSON output.")
        except Exception as e:
            log_msg(f"[Attempt {attempt}] An exception occurred during LLM call or parsing: {e}")
    
    log_msg("[ERROR] Failed to extract evidence after max retries.")
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset to extract evidence from external knowledge documents.")
    
    # Required positional arguments for input and output files
    parser.add_argument('input_file', type=str, help="Path to the input .jsonl file.")
    parser.add_argument('output_file', type=str, help="Path for the output .jsonl file.")

    # Required argument for external knowledge base
    parser.add_argument('--ek-base-path', type=str, required=True,
                        help="Base directory path for external knowledge (.md) files.")

    # Model input required (no default)
    parser.add_argument('--model', type=str, required=True,
                        help="Model name to use for evidence extraction.")

    # Default log path in current script directory
    default_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "Knowledge_Compression_log.log")
    parser.add_argument('--log-path', type=str, default=default_log_path,
                        help="Path for the output log file. Default: Knowledge_Compression_log.log")

    # Optional DB filter
    parser.add_argument(
        '--db-type',
        type=str,
        default='all',
        choices=['sqlite', 'bigquery', 'snow', 'all'],
        help="Specify the database type to process. (default: all)"
    )

    args = parser.parse_args()

    # --- Set global log path and model ---
    LOG_PATH = args.log_path
    model = args.model

    # Initial log records
    log_msg(f"Script started with the following configuration:")
    log_msg(f"  Input File: {args.input_file}")
    log_msg(f"  Output File: {args.output_file}")
    log_msg(f"  Log File: {args.log_path}")
    log_msg(f"  External Knowledge Path: {args.ek_base_path}")
    log_msg(f"  Model: {args.model}")
    log_msg(f"  Target Database Type: {args.db_type}")

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as out_f:
            for line in out_f:
                try:
                    record = json.loads(line.strip())
                    if "instance_id" in record:
                        processed_ids.add(record["instance_id"])
                except Exception:
                    continue
        log_msg(f"Loaded {len(processed_ids)} processed instance_ids from the output file.")

    with open(args.output_file, 'a', encoding='utf-8') as out_f:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log_msg(f"[Line {line_num}] JSON decoding error: {e}")
                    continue

                instance_id = record.get("instance_id")
                db_id = record.get("db") or record.get("db_id")

                # Filter records by DB type
                if args.db_type != 'all':
                    actual_db_type = detect_db_type(db_id)
                    if actual_db_type != args.db_type:
                        log_msg(f"[Line {line_num}] Skipping instance_id={instance_id} (db_type: {actual_db_type})")
                        continue

                log_msg(f"[Line {line_num}] Processing instance_id={instance_id} (db_id: {db_id})")
                
                if instance_id in processed_ids:
                    log_msg(f"[Line {line_num}] Skipped duplicate instance_id={instance_id}")
                    continue

                instruction = record.get("instruction") or record.get("question") or record.get("query") or record.get("input")
                external_knowledge = record.get("external_knowledge")
                evidence = ""

                if external_knowledge:
                    try:
                        EK = read_md_file(base_path=args.ek_base_path, md_filename=external_knowledge)
                        if EK:
                            evidence = extract_evidence(EK=EK, question=instruction, model=model)
                    except Exception as e:
                        log_msg(f"[Line {line_num}] Exception in evidence extraction for {instance_id}: {e}")
                        evidence = ""
                else:
                    log_msg(f"[Line {line_num}] No external knowledge for instance_id={instance_id}")

                record["evidence"] = evidence
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                log_msg(f"[Line {line_num}] Wrote instance_id={instance_id}\n")


'''
python your_script_name.py \
    /path/to/your/input.jsonl \
    /path/to/your/output.jsonl \
    --ek-base-path /path/to/your/resource/documents \
    --log-path /path/to/your/process.log \
    --db-type sqlite
'''