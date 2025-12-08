#!/bin/bash

# ========== Configuration Parameters ==========
# File path parameters
INPUT_FILE="spider2-lite/spider2-lite.jsonl"
OUTPUT_FILE="spider2-lite/spider2-lite_evidence.jsonl"

# Other command-line parameters
EK_BASE_PATH="spider2-lite/resource/documents"
MODEL_NAME="deepseek-chat"
DB_TYPE="all"

# ========== Execute Command ==========
python -m utils.preprocessor.Extract_evidence \
  "$INPUT_FILE" \
  "$OUTPUT_FILE" \
  --ek-base-path "$EK_BASE_PATH" \
  --model "$MODEL_NAME" \
  --db-type "$DB_TYPE"