Considering that most current tasks are centered around SQLite, and the SQL generation capabilities of open-source language models on Snowflake and BigQuery engines are rather limited, we hereby provide a comprehensive presentation of DSR-SQL's performance across various model sizes on the Spider-SQLite segment. **Note: The SQLite portion does not represent the full scope or difficulty of Spider 2.0.**

The table below presents the single-path generation performance of DSR-SQL when integrated with various types of LLMs. For the open-source models, [DeepSeek-V3.2-Exp was utilized for schema linking](../DSR_Lite/data/Spider2Lite_Sqlite.json), whereas Gemini completed the entire process as a standalone LLM.
> The results clearly indicate that the DSR-SQL framework is better suited for reasoning models (e.g., QwQ outperformed Qwen3Coder, even though Qwen3Coder is generally considered a more powerful model).

![SQLiet_Results](/assets/Sqlite_results.png)