#Todo Complete the parallel, sampling code
#model is the LLM required for SQL generation, which needs to be slightly more powerful
#Tool_model is the LLM for extracting the corresponding tables and columns (we have not yet found a Python native method to identify the corresponding tables and columns for Snowflake and Bigquery dialects)

#!/bin/bash

# ========== Configuration Parameters ==========

# 1. File path parameters (Relative to DSR_Lite root)
INPUT_FILE="spider2-lite/spider2-subset_lite_evidence.jsonl"
OUTPUT_FILE="spider2-lite/spider2-lite_SL.json"

# 2. Model parameters
MAIN_MODEL="Qwen/Qwen3-Coder-480B-A35B-Instruct"

TOOL_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"

# ========== Execute Command ==========
python -m utils.SL.Get_SL \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --model "$MAIN_MODEL" \
  --Tool_model "$TOOL_MODEL"