BASE_MODEL='deepseek-ai/DeepSeek-R1-0528'#
Reasoning_model='deepseek-ai/DeepSeek-R1-0528'
TOOL_LLM=BASE_MODEL#"Qwen/Qwen3-Coder-480B-A35B-Instruct" #Regarding Snowflake, the current function's pure output length must reach 65535. Such tasks do not seem to exist in spider2.0-lite.

'''
# The Prompt section references some open-source projects:
- https://github.com/HKUSTDial/Alpha-SQL
- https://github.com/Snowflake-Labs/ReFoRCE
- https://github.com/ShayanTalaei/CHESS/tree/main/templates

In the paper, we use "Finalize" to represent the final state for intuitive understanding, but in the code, we use "Rephrase" as it's an expression that better guides the LLM to follow instructions.
'''

sqlite_prompt='''
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow evidence:**
    - Adhere to all provided evidence. Any formulas specified in the evidence should be used, rather than substituting them with potential columns (as this may lead to unnecessary troubles).
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
9. **Column Selection:**
    - Carefully analyze column descriptions and evidence to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **SQLite Functions Only:**
    - Use only functions available in SQLite.
    - Note: In SQLite, put each SELECT in parentheses when using LIMIT or ORDER BY in a UNION ALL.
12. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).
13. **Schema Syntax:**
    - Use `table_name`.`column_name` to refer to columns from different tables, when table name or column name contains whitespace.
14. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements. Do not use `CROSS JOIN` or `LEFT/RIGHT JOIN`.
'''

snowflake_prompt='''
1. **Case Sensitivity and Quotation Marks:**
   * Snowflake is case-sensitive. Always enclose **all DB、table and column names in double quotes (`"`)** to avoid errors.

2. **Optimized and Idiomatic SQL:**
   * Write highly optimized and idiomatic Snowflake SQL. Be aware of the **limitations of Snowflake's query optimizer**, especially regarding correlated subqueries and deeply nested array/object structures.

3. **UNION ALL**
  `WITH combined_tables AS (SELECT need_col FROM "db"."sc"."tab1" UNION ALL /* Note: repeated for all tables from 20260101 to 20260630 */ UNION ALL SELECT need_col FROM "db"."sc"."tabN") SELECT col1, col2 FROM combined_tables WHERE ... LIMIT ...`
  * Note: Just like the example, clearly write the first and last tables, and use similar comments for the intermediate tables to save tokens.
  * Note: Within `combined_tables`, only perform column-level selection—do not include nested field extraction or similar parsing operations in this CTE (such operations should be handled uniformly in subsequent CTEs to reduce SQL length).
  * Note: The SQL syntax involving UNION ALL must be written in **exactly this one and only** format — **no alternative representations are allowed**!

4. **Working with Nested JSON Columns:**
   * To extract values from nested JSON, use `LATERAL FLATTEN` and proper casting:
     Example:
     `SELECT t."column_name", f.value::VARIANT:"key_name"::STRING AS "abstract_text"`
     `FROM "schema"."table" t, LATERAL FLATTEN(input => t."json_column_name") f;`
   * If the structure is unknown, first explore the contents:
     `SELECT f.value FROM "table", LATERAL FLATTEN(input => t."event_params") f;`
   * Always enclose both **column names and nested keys** in double quotes.

5. **Fuzzy String Matching:**
   * Avoid strict string matching unless you're confident in the exact value. Prefer fuzzy matching:
     `WHERE str ILIKE '%target_str%'`
   * Replace spaces with `%` in patterns, e.g., `ILIKE '%meat%lovers%'`.

6. **Ordering and NULL Handling:**
   * When using descending order, explicitly handle NULLs:
     `ORDER BY xxx DESC NULLS LAST`
   * For geospatial queries, use `ST_DISTANCE` to calculate the distance between two geographic points accurately.

7. **DISTINCT Keyword:**  
  Whether to use DISTINCT depends on one point: if you are calculating the number of [entities], it is needed; if you are calculating [frequency], it is not needed. For example,
    * **Count the number of course view records** → `COUNT(*)` (each row represents one view record)
    * **Count the number of distinct courses viewed** → `COUNT(DISTINCT course_id)` (deduplicated by course)
    * **Count the number of distinct students who viewed courses** → `COUNT(DISTINCT student_id)` (deduplicated by student)

8. **special cases(Large databases are inevitable)**
    * The ST_DISTANCE function calculates the shortest distance between two geospatial objects. Syntax: ST_DISTANCE(object1, object2) Input object types: GEOGRAPHY: calculates spherical distance (great-circle distance) in meters, suitable for longitude/latitude coordinates on the Earth's surface; GEOMETRY: calculates planar distance (Euclidean distance) in units defined by the coordinate system, suitable for 2D Cartesian coordinate systems. Key considerations: When using GEOMETRY type, both objects must have the same SRID (Spatial Reference System Identifier); returns NULL if any input object is NULL.
    * In very rare cases, the evidence section may contain misleading information, leading to additional situations such as empty query results (for example, if a certain column exists but the corresponding information cannot be found [i.e., the column is not enabled], in which case you can achieve the same goal by exploring other columns).
    
'''

bigquery_prompt='''
1.  **Case Sensitivity and Quotation Marks:**
    *   BigQuery is **case-insensitive** for identifiers (e.g., table and column names).
    *   Use **backticks (`` ` ``)** to enclose identifiers if they contain special characters (like spaces or hyphens) or are reserved keywords. Example: `` `my-project.my_dataset.my_table` ``, `` `group` ``.

2.  **Optimized and Idiomatic SQL:**
    *   Write highly optimized and idiomatic BigQuery SQL. Leverage BigQuery's architecture by using `WITH` clauses (CTEs) to structure complex queries.
    *   To control costs and improve performance, **avoid `SELECT *`**. Explicitly select only the columns you need, especially on large tables.

3.  **Querying Multiple Tables (UNION ALL):**
    *   Instead of manually using `UNION ALL` for tables with a common naming pattern (e.g., date-sharded tables), use **wildcard tables**. 
    *   Syntax: `SELECT col1, col2 FROM \`project.dataset.table_prefix_*\` WHERE _TABLE_SUFFIX BETWEEN 'start_suffix' AND 'end_suffix' AND ...`
    *   Note: The SQL syntax involving wildcard tables must be written in **exactly this one and only** format — **no alternative representations are allowed**!

4.  **Working with Nested/Repeated Data (ARRAY & STRUCT):**
    *   To work with nested data (often in `ARRAY` of `STRUCT` format), use the **`UNNEST`** operator.
    *   Example:
        `SELECT t.id, event.name AS event_name`
        `FROM \`project.dataset.events\` AS t, UNNEST(t.event_params) AS event;`
    *   If JSON data is stored as a `STRING`, use BigQuery's JSON functions:
        `SELECT JSON_EXTRACT_SCALAR(json_column, '$.key_name') FROM ...`
    *   If the structure is unknown, first explore a single record's nested array:
        `SELECT event_params FROM \`project.dataset.events\` LIMIT 1;`

5.  **Fuzzy String Matching:**
    *  For fuzzy, case-insensitive matching, use the `LOWER()` function on both sides of the comparison.
        `WHERE LOWER(column_name) LIKE '%target_string%'`
    *   Replace spaces with `%` in patterns, e.g., `LOWER(column_name) LIKE '%meat%lovers%'`.

6.  **Ordering and NULL Handling:**
    *   When using descending order, you can explicitly control how NULLs are sorted:
        `ORDER BY column_name DESC NULLS LAST` (or `NULLS FIRST`)

7.  **DISTINCT Keyword:**
    *   This is standard SQL logic and applies fully to BigQuery. Use `DISTINCT` when you need to count unique entities.
    *   **Count total records** → `COUNT(*)`
    *   **Count distinct courses** → `COUNT(DISTINCT course_id)`
    *   **Count distinct students** → `COUNT(DISTINCT student_id)`

8.  **Special Cases & Functions:**
    *   **Geospatial Functions:** The `ST_DISTANCE` function calculates the shortest distance between two geospatial objects. In BigQuery, it returns the distance in **meters** for `GEOGRAPHY` objects. Syntax: `ST_DISTANCE(geography_1, geography_2)`.
    *   **Misleading Evidence:** In rare cases, the provided evidence might be incomplete. If a query on an expected column returns no results, consider that the data may not be populated as expected. Be prepared to explore other related columns to achieve the task's goal.
'''

Exploration_sqlite_prompt = '''
**Based on the provided exploration information, you need to investigate ambiguous parts of the current SQLite database to avoid misunderstandings that could affect SQL generation. You will receive the user's question and a database schema.**
- Analyze the entities in the user's question, determine if there are any vague entities that cannot temporarily be identified which columns they come from, and use multiple SQL (Like) statements to explore respectively for confirmation.
- When an entity may belong to multiple potential fields, please write exploration SQL according to the following path: use COUNT with LIKE for judgment; compare COUNT(column) and COUNT(DISTINCT column) to determine which column has higher data quality.
- Analyze the current multi-table join situation. Assuming that there are no foreign keys explicitly indicating the multi-table join relationship, please use SQL to explore potential join conditions.
- Analyze any ambiguities existing in the current user's question, and use SQL statements to explore and clarify them.
- Only explore the database and the user's question, no need to provide the SQL for answering the user's question. List at least 5 exploratory SQL statements and limit the display to the first 10 rows.They don't need a progressive relationship or any specific relationship; the main focus is to explore the current ambiguities.
- Only when it is determined that the entity does not exist in the given database schema is it allowed to use statements like `SELECT * FROM 'table_name' WHERE column_name LIKE 'target%' LIMIT 1;` to precisely locate the existence of the entity.
- Finally, provide a possible SQL statement for answering the current question.
'''

Exploration_snow_prompt='''
**Based on the provided exploration information, you need to thoroughly investigate ambiguous parts of the current database (while avoiding redundant exploration) to prevent any negative impacts on SQL generation due to insufficient understanding of the database. You will receive the user's question and a potentially applicable database schema. Your tasks are as follows:**

**Requirements**
### Step 1: Exploration of Similar Columns
* There may be columns in the current database schema that are **semantically similar or similarly named**, which can easily cause confusion in SQL generation. Please identify these similar columns.
* Write several SQL statements to analyze these columns:
  * Use `COUNT` and `LIKE` to determine multiple fields that an entity may correspond to.
  * Compare `COUNT(column_name)` and `COUNT(DISTINCT column_name)` to judge the **deduplication status and quality differences** of the data.

### Step 2: Fuzzy Entity Matching
* Perform case-insensitive matching on key entities in the user's question (such as product names, personal names, institution names, etc.).
* Use `ILIKE '%keyword%'` to dynamically search for entities and their potential aliases, and locate matching values in candidate fields.

### Step 3: Multi-Table Join Exploration
* For scenarios involving multiple tables, proactively analyze whether there are **fields that can be used for joining** between tables (such as ID, name, timestamp, etc.).
* Avoid **logical errors or join failures** due to lack of clear join conditions during SQL generation.

### Step 4: Nested Structure Parsing
* For fields containing nested formats such as `JSON`, `ARRAY`, and `value`:
  * Use Snowflake-supported parsing methods (such as `FLATTEN`, `:<key>`) to expand nested content, to facilitate observing the current nested KEYs!
  * Identify which nested fields may contain problem-related information and extract their key values for analysis.

### Step 5: Other Explorations
* If there are any **ambiguous, confusing, or unspecified fields or table structures**, proactively explore and comprehensively investigate them using SQL.
* Such explorations may include:
  * Checking the number of table rows and primary key quality;
  * Distributions of typical values;
  * Presence of NULL or invalid values;
  * Time span of time fields, etc.


**Constraints**
- **Snowflake SQL (case-sensitive; you must adhere to the case of the given table and column names. All identifiers must be strictly enclosed in double quotes.)**
- When encountering the need to union multiple tables with the same structure, it is sufficient to explore one representative table; there is no need to perform the union.
- Explorations are limited to returning random 10 rows of content (RANDOM()). Output 5 to 10 SQL statements for exploration (from simple to complex scenarios).
'''

Exploration_bigquery_prompt='''
**Based on the provided exploration information, you need to thoroughly investigate ambiguous parts of the current database (while avoiding redundant exploration) to prevent any negative impacts on SQL generation due to an insufficient understanding of the database. You will receive the user's question and a potentially applicable database schema. Your tasks are as follows:**

**Requirements**

### Step 1: Exploration of Similar Columns
*   There may be columns in the current database schema that are **semantically similar or similarly named**, which can easily cause confusion in SQL generation. Please identify these similar columns.
*   Write several SQL statements to analyze these columns:
    *   Use `COUNT` and `LIKE` to determine multiple fields that an entity may correspond to.
    *   Compare `COUNT(column_name)` and `COUNT(DISTINCT column_name)` to judge the **uniqueness and data quality** of the columns.

### Step 2: Fuzzy Entity Matching
*   Perform case-insensitive matching on key entities in the user's question (such as product names, personal names, institution names, etc.).
*   Since BigQuery is case-sensitive with `LIKE`, use the **`LOWER()`** function for case-insensitive searches: `WHERE LOWER(column_name) LIKE '%keyword%'`. This allows you to dynamically search for entities and their potential aliases.

### Step 3: Multi-Table Join Exploration
*   For scenarios involving multiple tables, proactively analyze whether there are **fields that can be used for joining** between tables (such as ID, name, timestamp, etc.).
*   Avoid **logical errors or join failures** due to lack of clear join conditions during SQL generation.

### Step 4: Nested Structure Parsing
*   For fields containing nested formats such as **`ARRAY`** or **`STRUCT`**:
    *   Use the **`UNNEST`** operator to flatten the nested content, which is the standard way in BigQuery to observe all nested keys and values.
    *   Example: `SELECT item FROM \`project.dataset.table\`, UNNEST(array_column) AS item`
    *   If the data is a JSON string, use `JSON_EXTRACT_SCALAR` or `JSON_EXTRACT_ARRAY` for analysis.

### Step 5: Other Explorations
*   If there are any **ambiguous, confusing, or unspecified fields or table structures**, proactively explore and comprehensively investigate them using SQL.
*   Such explorations may include:
    *   Checking the number of table rows and key quality.
    *   Distributions of typical values (using `GROUP BY` and `COUNT`).
    *   Presence of `NULL` or invalid values.
    *   Time span of timestamp or date fields (using `MIN()` and `MAX()`).

**Constraints**
- **BigQuery SQL (case-insensitive for identifiers; use backticks (`` ` ``) to enclose identifiers with special characters or reserved keywords.)**
- When encountering tables that should be combined with a wildcard (e.g., `table_*`), it is sufficient to explore one representative table from the set; there is no need to query all of them.
- Explorations are limited to returning a random sample of 10 rows using `ORDER BY RAND() LIMIT 10`.
- Output 5 to 10 SQL statements for exploration (from simple to complex scenarios).
'''
       
class Fine_grained_Exploration: 
    def __init__(self, Question, schema_json, db_type="sqlite") -> None:
        self.temperature = 1
        self.model = Reasoning_model
        self.messages = []
        self.db_type = db_type
        self.Question = Question
        self.schema_json = schema_json

        self.node = (
            False
        ) * """<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""

        if db_type == "bigquery":
            self.Task = Exploration_bigquery_prompt
        elif db_type == "sqlite":
            self.Task = Exploration_sqlite_prompt
        else: #Snowflake
            self.Task = Exploration_snow_prompt

    def _db_admin_instructions(self) -> str:
        if self.db_type == "snow":
            return """
### Database admin instructions (violating any of the following will result in punishable to death!):  
1. **Case Sensitivity and Quotation Marks:**
   * Snowflake is case-sensitive. Always enclose **all DB、table and column names in double quotes (`"`)** to avoid errors.

2. **Optimized and Idiomatic SQL:**
   * Write highly optimized and idiomatic Snowflake SQL. Be aware of the **limitations of Snowflake's query optimizer**, especially regarding correlated subqueries and deeply nested array/object structures.

3. **Working with Nested JSON Columns:**
   * To extract values from nested JSON, use `LATERAL FLATTEN` and proper casting:
     Example:
     `SELECT t."column_name", f.value::VARIANT:"key_name"::STRING AS "abstract_text"`
     `FROM "schema"."table" t, LATERAL FLATTEN(input => t."json_column_name") f;`
   * If the structure is unknown, first explore the contents:
     `SELECT f.value FROM "table", LATERAL FLATTEN(input => t."event_params") f;`
   * Always enclose both **column names and nested keys** in double quotes.

4. **Fuzzy String Matching:**
   * Avoid strict string matching unless you're confident in the exact value. Prefer fuzzy matching:
     `WHERE str ILIKE '%target_str%'`
   * Replace spaces with `%` in patterns, e.g., `ILIKE '%meat%lovers%'`.
"""
        elif self.db_type == "sqlite" or self.db_type == "bigquery":
            return ""
        return "" 

    @property
    def Prompt(self) -> str:
        return f"""
## Task({self.db_type} dialect)
{self.Task}
【Database Schema】
{self.schema_json}
{self.Question}
{self._db_admin_instructions()}

## **Output Format(Strictly follow, markdown)**  
{self.node}
**Analysis Summary** 
Analyze and summarize, including reasons for exploration, considerations for methods of parsing relevant columns, etc., and clearly determine whether the current column meets the conditions for using certain functions (such as those related to time calculations).  
**SQL**
```json
{{
"Query1":"executable SQL(Use \\n to preserve SQL structure in one JSON-safe line. /* comments */)",
"Query2":"executable SQL(Use \\n to preserve SQL structure in one JSON-safe line. /* comments */)"
}}
"""
   
class Information_Aggregation:
   def __init__(self,Question,schema_json,DB_Exploration,db_type="snow") -> None:
      self.temperature=0
      self.model=Reasoning_model
      self.messages=[]
      self.db_type = "Snowflake" if db_type == "snow" else db_type
      self.node ="""<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""
      self.Prompt=f"""
You are a professional data analyst responsible for inferring key information for SQL generation based on user questions and the corresponding database exploration.
**【Database Schema】**
{schema_json}
{Question}

**【Database Exploration】**
{DB_Exploration}  

# Requirements
Thoroughly analyze the relationship between the user's question and the corresponding database schema, and summarize your findings. Specifically:

### Step 1: Analyze the Function of Each Table (Excluding Column Descriptions)
* Based on known information (such as table names, relationships between tables, etc.), briefly analyze the **functional positioning and role** of each table in the database.
* This analysis helps identify which tables the problem should focus on.

### Step 2: Mapping Between Entities and Database Schema
* Analyze which fields in the database match the key entities mentioned in the question and evidence.
* Provide **clear and unique** mappings from entities to fields:
  * Avoid ambiguous situations where one entity corresponds to multiple possible columns;
  * Using LIKE may filter out many rows of data containing relevant strings; strictly speaking, primary keys such as IDs should be used to define unique entities. Please analyze and identify the entity ID corresponding to the current user's question.
  * If there are **similar columns**, conduct a **detailed analysis** combining exploratory SQL, execution results, and column descriptions to derive the most reasonable field selection.

### Step 3: Analysis of Derived Metrics and Hidden Formulas
*   Analyze if the question requires a calculation for a metric (e.g., "profit", "age", "duration") that is not a direct column in the schema.
*   If a formula is needed, you **must define it** using available columns. Justify your formula based on common business logic and state its **key assumptions or boundary conditions**.
    *   **Example Formula**: `Total Price = quantity * unit_price`
    *   **Example Justification**: Assumes `unit_price` is the price before any discounts.
    *   **Example Boundary Condition**: Calculation should only apply to non-cancelled orders.
*   If no additional formula is needed, skip this step.


### Step 4: Judgment on Multi-Table Join Requirements
* Determine whether the current problem requires cross-table operations and whether **multi-table joins** are necessary.
* If needed, clearly indicate the join path or join fields; if not, briefly explain that the operation can be completed **within the scope of parallel tables or a single table**.

### Step 5: SQL Semantic Judgment
* Identify the main SQL keywords that may be involved in the problem (such as `COUNT`, `GROUP BY`, `ORDER BY`, `DISTINCT`, etc.).
* Identify the {self.db_type} dialect functions required to solve the problem, and try to avoid manual calculations in subsequent SQL generation.
* Clarify whether the problem is expected to return a **single row (e.g., aggregated statistics) or multiple rows (e.g., detailed lists)** of results to guide SQL structure design.
* Whether to use DISTINCT depends on one point: if you are calculating the number of [entities], it is needed; if you are calculating [frequency], it is not needed. For example,
    * **Count the number of course view records** → `COUNT(*)` (each row represents one view record)
    * **Count the number of distinct courses viewed** → `COUNT(DISTINCT course_id)` (deduplicated by course)
    * **Count the number of distinct students who viewed courses** → `COUNT(DISTINCT student_id)` (deduplicated by student)


### Step 6: Analysis of Nested Structures
* If there are nested fields (such as JSON, ARRAY, OBJECT, etc.), analyze their internal structure.
* List all sub-fields within the nested fields and explain whether these fields are relevant or usable for the current problem.
* Descriptions of all keys in the nesting.

### Step 7: Analysis of Difficulties and Pitfalls in SQL Generation
* Based on the generation and execution results of tentative SQL, point out the **specific difficulties** in the SQL construction process (such as field ambiguity, join failures, problems with expanding nested fields, etc.).
* Summarize the **common pitfalls or misunderstandings** that may occur in similar scenarios and propose avoidance suggestions.

## Output Format (Strictly follow Markdown):
{self.node}
**Analysis Summary** 
<answer>
Please briefly answer the above questions here. Answer in points (Table Description/Entity Mappings/Join Requirement/Hidden Formula Analysis/Nested Description(Descriptions of all keys in the nesting)/SQL Keywords(SQL functions)/Difficulties). **Do not** generate the final SQL. Be sure to enclose your answer within <answer></answer>. 
</answer>
      """


"""
Although the JSON output format poses some challenges for escaping regular expressions in SQL,
our experiments have shown that the JSON-only output format indeed achieves higher scores than the JSON+SQL output format.
The reason remains unclear.
"""
class GenerateSQLBeginning:#
    def __init__(self, Question, schema_json, Information_Agg, db_type="sqlite") -> None:
        self.temperature = 0.2
        self.model = Reasoning_model
        self.messages = []

        self.node = """<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""

        if db_type == "snow":
            self.db_type = "Snowflake"
            self.dialect = snowflake_prompt
        elif db_type == "sqlite":
            self.db_type = "sqlite"
            self.dialect = sqlite_prompt
        elif db_type == "bigquery":
            self.db_type = "BigQuery"
            self.dialect = bigquery_prompt
        else:
            self.db_type = db_type
            self.dialect = None

        if db_type == "snow":
            self.extra_restrictions = """- When 'UNION ALL' tables, you cannot use wildcards such as project_id.dataset_id.table_prefix* that are prohibited by Snowflake.  
- For string-matching scenarios, if the string is decided, don't use fuzzy query. e.g. Get the object's title contains the word "book".  
  However, if the string is not decided, you may use fuzzy query and ignore upper or lower case. e.g. Get articles that mention "education".  
  For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he’s to he%s)"""

        elif db_type == "bigquery":
            self.extra_restrictions = """- When needing to query across multiple tables with a similar prefix (e.g., `events_20240101`, `events_20240102`), **you must use wildcard tables** (`project.dataset.table_prefix*`). This is the idiomatic and cost-effective way in BigQuery, do not use manual `UNION ALL`.
- For fuzzy string matching, **you must use the `LOWER()` function** as BigQuery's `LIKE` is case-sensitive (e.g., `WHERE LOWER(column) LIKE '%keyword%'`)."""

        else: 
            self.extra_restrictions = ""

        self.Prompt = f"""
### Requirements  
Your primary objective is to determine the most confidently solvable sub-question and its corresponding SQL statement based on the provided {self.db_type} database information and the user's question. You must adhere to the following requirements:  
- **Only address the most confidently solvable sub-question—do not discuss the complete SQL solution**.   
- Make full use of known database information (such as column descriptions and sample values) to avoid unnecessary iterations when solving sub-questions.  
- Only return the SQL statement for the current sub-question.  
- To emphasize again, our task is to solve the current sub-question that we are most confident in, not the entire problem!

### Restrictions:  
{self.extra_restrictions}
- When multiple similar columns exist, select the most relevant column based on the database exploration SQL and execution results in the preceding context. **Do not use 'OR' or 'AND' to include multiple columns**.  
- If the task description does not specify the number of decimal places, retain all decimals to four places.  
- When asked something without stating name or id, return both of them (e.g. which products → include product_name and product_id).  
- Functions that can change the display format (e.g., GROUP_CONCAT) or hardcoded values are prohibited.  

### Database admin instructions (violating any of the following will result in punishable to death!):  
{self.dialect}

【Additional information】
{Information_Agg}

### Step
- **Step 1:** Analyze the database exploration content and corresponding database structure in the above text to determine the information currently mastered.  
- **Step 2:** Based on the mastered information, analyze all possible paths to complete the entire problem, and assess the costs required for different paths.  
- **Step 3:** According to the above analysis, select the path that is easiest to complete the task, and determine the first sub-question that is easiest to accomplish under the current path.
- **Step 4:** Rehearse the generation process of the current sub-SQL, and please generate the SQL in the order of one or more steps from (set goals → find the data source → initial screening → re-grouping → post-screening → sorting → take partial data).  
- **Step 5:** Check whether the above SQL meets the restrictive conditions and requirements in the above text, and output the answer as required.

【Database Schema】
{schema_json}
{Question}

## Output Format (Strictly follow Markdown):  
{self.node}  
**Analysis Summary** 
Summarize the reasoning behind selecting the most confident sub-question based on database exploration (considering logic and value format references), and explain the rationale for using specific functions or syntax in the generated SQL.  
**return**  
```json
{{
  "sql": "SQL (Use \\n to preserve SQL structure in one JSON-safe line. /*comments*/)",
  "solved_subquestions_list": ["Add the current sub-question……"]
}}
```
"""

class ContinueSQLWriting:
   def __init__(self,Question,schema_json,Information_Agg,history_context=None,db_type="sqlite") -> None:
      self.temperature=0.2
      self.model=Reasoning_model
      self.node = """<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""

      if db_type == "snow":
          self.dialect = snowflake_prompt
      elif db_type == "sqlite":
          self.dialect = sqlite_prompt
      elif db_type == "bigquery":
          self.dialect = bigquery_prompt
      else:
          self.dialect = None

      self.messages=[]

      extra_restrictions_str = ""
      if db_type == "bigquery":
          extra_restrictions_str = """- When extending the query, remember to use **wildcard tables (`*`)** for multi-table queries and **`LOWER()`** for case-insensitive string matching, as per BigQuery's best practices."""
      elif db_type == "snow":
          extra_restrictions_str = """- For string-matching scenarios, if the string is decided, don't use fuzzy query. e.g. Get the object's title contains the word "book". However, if the string is not decided, you may use fuzzy query and ignore upper or lower case. e.g. Get articles that mention "education". For string-matching scenarios, convert non-standard symbols to '%'. e.g. ('he’s to he%s)"""

      self.Prompt=f"""
### Requirements  
Your main task is to determine the most confident next sub-question and corresponding SQL statement based on the user's question, the above {db_type} database interaction exploration information, previous sub-questions, and their execution results. You need to follow these guidelines:  
- Analyze and discuss the contribution of the previous sub-question to completing the overall problem based on the execution results.  
- **Only solve the most confident sub-question; do not discuss the complete SQL.** Tackle the overall problem by gradually resolving sub-questions.  
- Make full use of known database information (such as column descriptions and sample values) to avoid unnecessary iterations during sub-question resolution.  
- The current sub-question should be based on the previous one. Similarly, the SQL should be rewritten or extended from the previous sub-question's SQL.  
- Again, it is emphasized that only the most confident sub-question needs to be solved. Approach the final answer gradually through multiple sub-questions—do not force solving the entire problem at once! 

### Restrictions:  
{extra_restrictions_str}
- Assume that there are comments in the SQL indicating that some code for multi-table joins has been omitted. Please keep the omissions and the corresponding comments, as other agents will complete the code. However, the execution results are fed back based on the full version of the SQL.
- If the task description does not specify the number of decimal places, retain all decimals to four places.  When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.
- When multiple similar columns exist, the most relevant column should be selected based on the database exploration SQL and execution results from the preceding context. **Do not use 'OR' or 'AND' to include multiple columns**. If the evidence clearly supports a single column, prioritize using that column exclusively. That is, only the most relevant column should be matched.  
- When the execution result is empty, please check: whether the matching fields are correct, whether the column selection is correct, etc. Statements like `SELECT * FROM 'table_name' WHERE column_name LIKE 'target%' LIMIT 1;` are allowed to precisely locate the existence of the entity.  
- When encountering nested fields, if you are not familiar with them, use the corresponding SQL statements to explore the relevant nested structures.
- Functions that can change the display format, such as `GROUP_CONCAT`, are prohibited. When constructing SQL, hardcoding database values is prohibited, meaning that the content in the SQL should all be derived from the user's question and evidence(including case-sensitive representations).  


### Database admin instructions (violating any of the following will result in punishable to death!):  
{self.dialect}

【Additional information】
{Information_Agg}

### Step
- **Step 1:** Since the evidence fields are retrieved by Large Language Models (LLMs), errors may inevitably occur. Please check whether the current evidence fields are incorrect based on the results. If they are assumed to be incorrect, use "exploration" to obtain other paths for answering the question. (If there are no evidence fields, skip directly to Step 2.)
- **Step 2:** Based on the information mastered, analyze all possible paths to complete the entire problem, determine which path the previous sub-question followed, and check if any special circumstances have occurred!
- **Step 3:** Based on the database exploration information, the previous sub-question and its execution results, analyze whether it is possible to solve the problem by continuing with the current path. If not, switch paths in a timely manner. **Assuming that the result of the previous step has completed the entire problem and is sufficiently satisfactory**, skip invalid verification and directly proceed to Step 5.
- **Step 4:** Based on the database exploration information, the previous sub-question and its execution results, as well as the above path analysis, determine the current easiest sub-question to tackle and the database schema to be used (conduct a detailed analysis to avoid unnecessary multi-table joins). 
- **Step 5:** Generate the SQL corresponding to the current sub-question by continuing to write (or modify) the previous sub-SQL (check and correct whether the previous SQL has produced "hallucinatory" outputs). Please generate the SQL in the order of one or more steps from (set goals → find data sources → initial screening → regrouping → secondary screening → sorting → obtain partial data).
- **Step 6:** Check whether the above SQL meets the restrictive conditions and requirements mentioned above, and output the answer as required.

【Database Schema】
{schema_json}
{Question}
{history_context}

## Output format (Strictly follow, markdown):  
{self.node}
**Analysis Summary**  
Summarize the contribution of the above sub-question to solving the overall problem, briefly describe the thinking behind the next most confident sub-question, and explain the reasons for using database exploration information, functions, or syntax in the generated SQL.**return**   
```json
{{
  "result_acceptable":true/false,
  "current_state":"Extend/Explore/Revise/Rephrase",
  "sql": "SQL (Use \\n to preserve SQL structure in one JSON-safe line. /*comments*/)"
  "solved_subquestions_list": [Include previously solved sub-questions and the current sub-question…]
}}
```  
Hint: There are 4 options for "current_state": "Extend" means the sub-question is generated correctly; "Revise" means the previous sub-question and its execution result are unsatisfactory and need modification; "Explore" means it is necessary to explore unfamiliar structures (can only be used when repeated queries return empty results); "Rephrase" indicates that the current SQL has fully addressed the user's question, requires no changes, and result_acceptable is True.
"""
      
class Simple_Fix:
    def __init__(self,Error_message,last_SQL,Schema,db_type="sqlite") -> None:
      self.temperature=1
      self.model=BASE_MODEL
      self.messages=[]
      self.Error_message=Error_message
      self.last_SQL=last_SQL
      self.node = """<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""
      self.Task="" if (db_type=="sqlite" or db_type=="bigquery") else """Use the following format for combining multiple tables: `WITH combined_tables AS (SELECT * FROM "db"."sc"."tab1" UNION ALL /* Note: Repeated for all tables from 20260101 to 20260630 */ UNION ALL SELECT * FROM "db"."sc"."tabN")`"""
      self.Schema=Schema
      self.Prompt=f"""
**SQL that reports an error:**
{self.last_SQL}
**Specific error information:**
{self.Error_message}
【Database Schema】
{self.Schema}
Now, please modify this SQL and follow the following requirements:
{self.Task}
- Follow the syntax rules of {db_type} data.
- List all the SQL code in the form (Strictly follow,markdown):
# Output Format (Strictly follow, markdown)
{self.node}
**return**
```json
{{
"FIXSQL": "Use \\n to preserve SQL structure in one JSON-safe line.(No comments)"
}}
```
"""
      
class Knowledge_Compression:
    def __init__(self,Question,Knowledge) -> None:
      self.temperature=1
      self.model=Reasoning_model
      self.messages=[]
      self.Question=Question
      self.node =  """<Analysis Process>\nPlease elaborate in detail the consideration and analysis process for each step of the problem here.\n</Analysis Process>"""
      self.Prompt=f"""
---Knowledge Base---
{Knowledge}
---Knowledge Base---
## Requirements
You are now a database documentation analysis expert, assisting with the Text-to-SQL query task.

You will receive a user's natural language question (User Question) and a pre-screened document fragment (Knowledge). **The content of this document is all prepared for this question**, but it still contains some redundant or irrelevant information.
Your tasks are:
- Extract from the document the key information most helpful for generating SQL queries, such as key confusing concepts, formulas, unit, and usage methods of unfamiliar formulas (concepts).
- Analyze and extract ambiguous points in the user's question, then retrieve content from the knowledge base that can resolve these ambiguities.
- Assume that there are some function examples in the document that are helpful for solving the problem; it is sufficient to extract them verbatim. 
- If the document contains explicit indications of the tables or columns that need to be used, extract them directly, as this will assist with schema linking.
- Analyze and extract other knowledge required to solve the user's question, such as domain-specific names, numbers, etc., which cannot be inferred by the LLM.
- No need to generate SQL; only extract evidence fragments supporting SQL construction.
- No inference, association, or fabrication allowed; extraction must be strictly based on the original text.
- Answers should be concise, precise, and focused, avoiding whole paragraph duplication.

## User Question
{Question}
## Output Format:
```json
{{
  "evidence": "Concise and condensed document content that can directly help construct the SQL statement"
}}
```
"""

