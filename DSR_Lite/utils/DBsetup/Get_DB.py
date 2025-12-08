import json
import os
import shutil

def read_db_config():
    # Directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "DB.json")
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize return variables
    sqlite_path = ""
    snow_path = ""
    bigquery_path = ""
    snow_auth = ""
    bigquery_auth = ""
    
    # Iterate through configuration items
    for item in data:
        db_type = item.get("DB_type", "").lower()
        local_path = item.get("Local_path", "")
        auth = item.get("Authentication", "")
        
        if db_type == "sqlite":
            sqlite_path = local_path
        elif db_type == "snowflake":
            snow_path = local_path
            snow_auth = auth
        elif db_type == "bigquery":
            bigquery_path = local_path
            bigquery_auth = auth

    if sqlite_path and os.path.exists(os.path.dirname(os.path.normpath(sqlite_path))):
        # 1. Locate spider2-localdb path (in the same directory as sqlite_path)
        # normpath removes trailing slashes, dirname gets parent directory
        base_dir = os.path.dirname(os.path.normpath(sqlite_path))
        localdb_path = os.path.join(base_dir, 'spider2-localdb')

        # 2. Check if spider2-localdb folder exists; raise error if missing
        if not os.path.exists(localdb_path):
            raise FileNotFoundError(
                f"\n[Error] Missing 'spider2-localdb' folder at: {localdb_path}\n"
                f"Please download the required files: https://github.com/xlang-ai/Spider2/tree/main/spider2-lite#-quickstart"
            )

        # 3. Traverse each database subfolder under sqlite_path
        if os.path.exists(sqlite_path):
            for db_name in os.listdir(sqlite_path):
                db_folder = os.path.join(sqlite_path, db_name)

                # Ensure it's a folder (e.g., AdventureWorks)
                if os.path.isdir(db_folder):
                    target_sqlite_file = os.path.join(db_folder, f"{db_name}.sqlite")

                    # 4. Check if corresponding .sqlite file exists; copy from localdb if missing
                    if not os.path.exists(target_sqlite_file):
                        source_sqlite_file = os.path.join(localdb_path, f"{db_name}.sqlite")

                        if os.path.exists(source_sqlite_file):
                            print(f"[Auto-Fix] Missing sqlite file for '{db_name}'.")
                            print(f"           Copying from: {source_sqlite_file}")
                            print(f"           To: {target_sqlite_file}")
                            shutil.copy2(source_sqlite_file, target_sqlite_file)
                        else:
                            print(f"[Warning] SQLite file missing in both destination and spider2-localdb: {db_name}.sqlite")
    
    return sqlite_path, snow_path, bigquery_path, snow_auth, bigquery_auth

# Usage example
if __name__ == "__main__":
    sqlite, snow, bigquery, snow_auth, bigquery_auth = read_db_config()
    
    print(f"SQLite path: {sqlite}")
    print(f"Snowflake path: {snow}")
    print(f"BigQuery path: {bigquery}")
    print(f"Snowflake authentication: {snow_auth}")
    print(f"BigQuery authentication: {bigquery_auth}")