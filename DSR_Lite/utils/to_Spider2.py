import os
import json
import shutil
import argparse

def process_files(input_folder, output_folder):
    """
    Processes JSON files from an input folder, extracts SQL queries,
    and saves them to individual .sql files in an output folder.

    Args:
        input_folder (str): The path to the folder containing the input .json files.
        output_folder (str): The path to the folder where .sql files will be saved.
    """
    
    # Counter for processed records
    count = 0

    # If the output folder already exists, delete it first and then create it.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all .json files in the input_folder
    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        input_json_path = os.path.join(input_folder, filename)

        try:
            with open(input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {input_json_path}: {e}")
            continue

        # Iterate over each record, extract the SQL, and write it to a file
        for item in data:
            instance_id = item.get("instance_id")
            sql = item.get("final_SQL")

            if instance_id:
                output_file_path = os.path.join(output_folder, f"{instance_id}.sql")

                # If sql is None or an empty string, write a placeholder error message
                content = sql.strip() if isinstance(sql, str) else ""
                if not content:
                    content = "SELECT 'Workflow Error' AS result;"

                with open(output_file_path, "w", encoding="utf-8") as out_file:
                    out_file.write(content + "\n")  # Ensure the file ends with a newline

                print(f"[{filename}] Written: {output_file_path}")
                count += 1
            else:
                print(f"[{filename}] Skipping item due to missing instance_id: {item}")

    print(f"\nTotal SQL files written: {count}")

def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description="Extract SQL queries from JSON files and save them to individual .sql files."
    )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the directory containing the source JSON files."
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the directory where the output .sql files will be saved."
    )
    
    args = parser.parse_args()
    
    process_files(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
"""
python to_Spider2.py \
    --input_folder "outcome" \
    --output_folder "sql"
"""