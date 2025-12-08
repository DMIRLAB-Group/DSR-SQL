# Place data preprocessing code here, corresponding to the Schema and Knowledge Refinement section in the paper
#!/bin/bash

# ========== Configuration Parameters ==========
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"
WORKERS=8

# ========== Execute Commands ==========

echo "Starting SQLite processing..."
# 1. SQLite, no LLM required
python -m utils.preprocessor.Get_table_mes_sqlite


echo "Starting BigQuery processing..."
# 2. BigQuery, requires LLM
python -m utils.preprocessor.Get_table_mes_bigquery \
  --model "$MODEL_NAME"


echo "Starting Snowflake processing..."
# 3. Snowflake, takes slightly longer
python -m utils.preprocessor.Get_table_mes_snow \
  --model "$MODEL_NAME" \
  --workers "$WORKERS"

echo "All tasks completed."