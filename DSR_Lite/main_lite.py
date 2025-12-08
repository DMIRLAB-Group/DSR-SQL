import os
import sys
import json
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
import argparse

# Local imports
from utils.extract_json import *
from utils.Prompt import *
from utils.Database_Interface import *
from utils.app_logs.logger_config import setup_logger, log_context, JsonLogger
from LLM.LLM_OUT import *



def save_or_load_pickle(data=None, filename='data.pkl', mode='save'):
    """
    Saves or loads a pickle file in the specified directory.
    Parameters:
        data: The data object to be saved (only required when mode='save')
        filename (str): The filename (without path)
        mode (str): 'save' to save, 'load' to load
    Returns:
        If mode='load', returns the loaded data; otherwise returns None
    """
    # Construct the full path
    # Set the base path
    # Create temp_path if it doesn't exist
    if not os.path.exists(temp_path):
        os.makedirs(temp_path, exist_ok=True)
    file_path = os.path.join(temp_path, filename)

    if mode == 'save':
        if data is None:
            raise ValueError("The 'data' argument must be provided when mode='save'")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to {file_path}")

    elif mode == 'load':
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist, please confirm if it has been saved")
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"Data successfully loaded from {file_path}")
        return loaded_data

    else:
        raise ValueError("The 'mode' argument must be 'save' or 'load'")

#---- Schema-aware Alignment----

def Fine_grained_Exploration_func(Question_id,Question, schema_json, db_name, base_mess=[], step="Exploration Stage",db_type='sqlite'):
    log_msg(f"\n{'-'*40}【Question_id: {Question_id}】 | 【Start Stage: {step}】{'-'*40}")

    # Initialize fine-grained exploration module
    FGE = Fine_grained_Exploration(Question=Question, schema_json=schema_json,db_type=db_type)
    FGE_mess = base_mess + [{"role": "user", "content": FGE.Prompt}]
    log_msg(f"Prompt：{FGE_mess}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            log_msg(f"\n[Fine-grained Exploration] Attempting to call language model for the {attempt + 1} time...")
            input_token_count, output_token_count, Thinking, LLM_return = LLM_output(messages=FGE_mess,
                                        model=FGE.model,
                                        temperature=FGE.temperature
                                        )

            logger_status.log(
                question_id=Question_id,
                step=step,
                if_in_fix="NO",
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                status=None 
            )

            log_msg(f"\n[【Question_id: {Question_id}】 | Fine-grained Exploration] LLM Thinking content:\n{Thinking}")
            log_msg(f"\n[【Question_id: {Question_id}】 | Fine-grained Exploration] LLM output content:\n{LLM_return}")
            ge_sql = extract_and_parse_json(text=LLM_return)
            break
        except Exception as e:
            log_msg(f"[【Question_id: {Question_id}】 | Fine-grained Exploration Retry {attempt + 1}/{max_retries}] Error: {e}")
    else:
        log_msg(f"[【Question_id: {Question_id}】 | Fine-grained Exploration] Parsing failed, maximum retries reached, exiting.")
        return []

    query_list = []
    sql_list = list(ge_sql.values())

    for idx, original_sql in enumerate(sql_list):
        log_msg(f"\n{'='*20} [Executing Original SQL #{idx + 1}] {'='*20}")
        log_msg(f"[【Question_id: {Question_id}】 | Original SQL Statement]:\n{original_sql}\n")

        # Execute SQL
        status, result = db_interface(db_type=db_type, query=original_sql, conn_info=db_name)

        if status == 0:
            log_msg(f"[【Question_id: {Question_id}】 | SQL Execution Successful]\nResult:\n{result}")
            query_list.append({"role": "user", "content": original_sql})
            query_list.append({"role": "assistant", "content": "Execution result:\n" + result})
            continue

        # Start repair mechanism
        log_msg(f"\n{'-'*40}【【Question_id: {Question_id}】 | Initiating Repair Mechanism: {step} Repair Stage】{'-'*40}")
        fix_attempts = 0
        current_sql = original_sql
        accumulated_prompt = f"Original SQL:\n{original_sql}\nError Message:\n{result}\n"
        while fix_attempts < 5:
            SF = Simple_Fix(Error_message=result, last_SQL=current_sql, Schema=schema_json,db_type=db_type)
            fix_prompt = accumulated_prompt + "\n" + SF.Prompt

            sf_mess = base_mess + [{"role": "user", "content": fix_prompt}]
            log_msg(f"fix prompt: {sf_mess}")
            log_msg(f"\n[【Question_id: {Question_id}】 | Repair Attempt #{fix_attempts + 1}] Calling language model to fix SQL...")

            input_token_count, output_token_count, Thinking, LLM_return = LLM_output(messages=sf_mess,
                                        temperature=SF.temperature,
                                        model=SF.model
                                        )
            fix_statu = {"triggering_error": result}

            logger_status.log(
                question_id=Question_id,
                step=f"{step} Repair Stage",
                if_in_fix="YES",
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                status=fix_statu
            )
            
            log_msg(f"[【Question_id: {Question_id}】 |  Repair Stage LLM Thinking]:\n{Thinking}")
            log_msg(f"[【Question_id: {Question_id}】 |  Repair Stage LLM Output]:\n{LLM_return}")

            try:
                fixed_sql_dict = extract_and_parse_json(LLM_return)
                fixed_sql = list(fixed_sql_dict.values())[0]
                log_msg(f"[【Question_id: {Question_id}】 |  Parsed Fixed SQL]:\n{fixed_sql}")
            except Exception as e:
                log_msg(f"[【Question_id: {Question_id}】 |  SQL Repair Parsing Error] Parsing failed for the {fix_attempts + 1} time: {e}")
                fix_attempts += 1
                continue

            # Execute the fixed SQL
            status, result = db_interface(db_type=db_type, query=fixed_sql, conn_info=db_name)

            if status == 0:
                log_msg(f"[【Question_id: {Question_id}】 |  Repair Successful] Execution Result:\n{result}")
                query_list.append({"role": "user", "content": fixed_sql})
                query_list.append({"role": "assistant", "content": "Execution result:\n" + result})
                break
            else:
                log_msg(f"[【Question_id: {Question_id}】 |  Repair Failed] Failed for the {fix_attempts + 1} time, error message:\n{result}")
                accumulated_prompt += f"\nFixed SQL attempt {fix_attempts + 1}:\n{fixed_sql}\nError Message:\n{result}\n"
                current_sql = fixed_sql
                fix_attempts += 1

        if fix_attempts == 5:
            log_msg(f"\n[【Question_id: {Question_id}】 |  Maximum Repair Attempts Exceeded] Skipping current SQL.\nOriginal SQL:\n{original_sql}")

    log_msg(f"\n{'='*40}【【Question_id: {Question_id}】 |  {step} Stage End】{'='*40}\n")
    return query_list

def Information_Summary(Question_id,Question, schema_json, DB_Exploration, step="Summarization Stage"):
    log_msg(f"\n{'-'*40}【Question_id: {Question_id}】 |  Start Stage: {step}】{'-'*40}")
    db_exploration_str = "\n".join(str(d["content"]) for d in DB_Exploration) # Build into a string

    IA = Information_Aggregation(Question=Question, schema_json=schema_json, DB_Exploration=db_exploration_str)
    IA_mess = [{"role": "user", "content": IA.Prompt}]
    log_msg(f"【Question_id: {Question_id}】 |  LLM Input: {IA_mess}")
    max_attempts = 3
    for attempt in range(max_attempts):
        log_msg(f"\n[【Question_id: {Question_id}】 |  Information Aggregation Stage] Calling language model for the {attempt + 1} time...")

        input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
            messages=IA_mess,
            model=IA.model,
            temperature=IA.temperature
        )

        logger_status.log(
            question_id=Question_id,
            step=step,
            if_in_fix="NO",
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            status=None 
        )
        
        log_msg(f"[【Question_id: {Question_id}】 |  Language Model Thinking]:\n{Thinking}")
        log_msg(f"[【Question_id: {Question_id}】 |  Language Model Output]:\n{LLM_return}")
        
        try:
            return extract_answer_content(text=LLM_return)
        except ValueError as e:
            log_msg(f"[【Question_id: {Question_id}】 |  Parsing Failed] Failed to extract <answer> content for the {attempt + 1} time: {e}")
            if attempt == max_attempts - 1:
                log_msg(f"[【Question_id: {Question_id}】 |  Terminated] Maximum retries reached, still unable to extract <answer> tag content")
                raise  # Can choose to raise an exception, or return None or a default value

    return None  # Theoretically, this line should not be reached unless the loop encounters an error

#---- Schema-aware Alignment----

#---- Generation-State Evolution----

def GenerateSQL1(Question_id,Question, schema_json, db_name,Information_Agg, base_mess=[], db_type="sqlite", step="Initial SQL Generation Stage"):
    expected_keys = {
        "sql",
        "solved_subquestions_list"
    }

    log_msg(f"\n{'-'*40}【Question_id: {Question_id}】 |  Start Stage: {step}】{'-'*40}")

    GSB = GenerateSQLBeginning(Question=Question, schema_json=schema_json,Information_Agg=Information_Agg,db_type=db_type)
    GSB_mess = base_mess + [{"role": "user", "content": GSB.Prompt}]
    max_retries = 5
    log_msg(f"prompt: {GSB_mess}")
    for attempt in range(max_retries):
        try:
            log_msg(f"\n[【Question_id: {Question_id}】 |  {step}] Calling language model for the {attempt + 1} time...")

            # Modified: Added thinking parameters consistent with reference code
            input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                messages=GSB_mess,
                model=GSB.model,
                temperature=GSB.temperature
            )
            log_msg(f"[【Question_id: {Question_id}】 |  Language Model Thinking]:\n{Thinking}")
            log_msg(f"[【Question_id: {Question_id}】 |  Language Model Output]:\n{LLM_return}")
            
            statu = extract_and_parse_json(text=LLM_return)

            logger_status.log(
                question_id=Question_id,
                step=step,
                if_in_fix="NO",
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                status=statu,
                # SQL=statu.get("sql", "") 
            )
            
            log_msg(f"[【Question_id: {Question_id}】 |  Parsing successful, returned fields]: {set(statu.keys())}")

            if set(statu.keys()) == expected_keys:
                
                # Modified: Use statu["sql"] directly instead of raw_sql
                flag,current_subsql = SQL_completion(statu["sql"], db_type)
                log_msg(f"【Question_id: {Question_id}】 |  \n✅ SQL structure is valid, starting SQL execution:\n{current_subsql}")
                status, result = db_interface(db_type=db_type, query=current_subsql, conn_info=db_name)

                if status == 0:
                    log_msg(f"[【Question_id: {Question_id}】 |  SQL Execution Successful]\nResult:\n{result}")
                    if flag==1:
                        result=result
                    else:
                        result=result+"\n*The SQL remains in an abbreviated form, but the returned answer is generated from the full version of the SQL."
                    return [
                        {"role": "user", "content": str(statu)},
                        {"role": "assistant", "content": "Execution result:\n" + result}
                    ],statu,current_subsql
                else:
                    log_msg(f"【Question_id: {Question_id}】 |  \n⚠️ Initial SQL execution failed:\nError Message:\n{result}")
                    log_msg(f"{'-'*20}【【Question_id: {Question_id}】 |  Entering Repair Process】{'-'*20}")

                    fix_mess = GSB_mess + [
                        {"role": "assistant", "content": str(statu)},
                        {"role": "user", "content": result + "\nPlease analyze and fix the current " + db_type + 
                        " SQL error" + 
                        # Snowflake
                        (", and note that in certain cases, some columns in the current database do not support the use of double quotes. Please detect and handle such situations." if db_type == "snow" else "") + 
                        # BigQuery
                        (", and pay close attention to BigQuery's quoting rules: use backticks (`` ` ``) for identifiers (like `project.dataset.table`) and single/double quotes (' or \") for string values. Misusing quotes is a very common error." if db_type == "bigquery" else "") +
                        # ALL
                        ", and return the corrected SQL in the same Markdown JSON format (including the key names) with ```json```"}
                    ]

                    log_msg(f"fix prompt: {fix_mess}")
                    for fix_attempt in range(max_retries):
                        try:
                            log_msg(f"\n[【Question_id: {Question_id}】 |  Repair Attempt #{fix_attempt + 1}] Calling language model for repair...")

                            # Modified: Added thinking parameters consistent with reference code
                            input_token_count, output_token_count, Thinking, fix_return = LLM_output(
                                messages=fix_mess,
                                model=GSB.model,
                                temperature=GSB.temperature
                            )
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair LLM Thinking]:\n{Thinking}")
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair LLM Output]:\n{fix_return}")
                            
                            fix_statu = extract_and_parse_json(fix_return)
                            # Modified: Removed raw_sql=extract_sql(fix_return)

                            logger_status.log(
                                question_id=Question_id,
                                step=f"{step} Repair",
                                if_in_fix="YES",
                                input_token_count=input_token_count,
                                output_token_count=output_token_count,
                                status=fix_statu,
                                # SQL=fix_statu.get("sql", "")
                            )
                            
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair parsing successful, returned fields]: {set(fix_statu.keys())}")

                            if set(fix_statu.keys()) == expected_keys:
                                # Modified: Use fix_statu["sql"] directly
                                flag,current_subsql = SQL_completion(fix_statu["sql"], db_type)
                                log_msg(f"[【Question_id: {Question_id}】 |  Attempting to execute repaired SQL]:\n{current_subsql}")
                                status, result = db_interface(db_type=db_type, query=current_subsql, conn_info=db_name)

                                if status == 0:
                                    log_msg(f"[【Question_id: {Question_id}】 |  Repaired SQL Execution Successful]\nResult:\n{result}")
                                    if flag==1:
                                        result=result
                                    else:
                                        result=result+"\n*The SQL remains in an abbreviated form, but the returned answer is generated from the full version of the SQL."
                                    return [
                                        {"role": "user", "content": str(fix_statu)},
                                        {"role": "assistant", "content": "Execution result:\n" + result}
                                    ],fix_statu,current_subsql
                                else:
                                    log_msg(f"【Question_id: {Question_id}】 |  ⚠️ Repair attempt {fix_attempt + 1} failed:\n{result}")

                                    fix_mess += [
                                        {"role": "assistant", "content": str(fix_statu)},
                                        {"role": "user", "content": result + "\nPlease analyze and fix the current " + db_type + " SQL error again" +
                                        (", and in certain cases, some columns in the current database do not support the use of double quotes. Please detect and handle such situations" if db_type == "snow" else "") +
                                        (", and pay close attention to BigQuery's quoting rules: use backticks (`` ` ``) for identifiers (like `project.dataset.table`) and single/double quotes (' or \") for string values. Misusing quotes is a very common error." if db_type == "bigquery" else "") +
                                        ", and return the corrected SQL in the same Markdown JSON format (including the key names) with ```json```."}
                                    ]

                            else:
                                log_msg(f"【Question_id: {Question_id}】 |  ❌ Repair attempt {fix_attempt + 1} returned incorrect format: actual keys are {set(fix_statu.keys())}")

                        except Exception as e:
                            log_msg(f"【Question_id: {Question_id}】 |  ❌ Repair attempt {fix_attempt + 1} parsing failed: {e}")

                    log_msg(f"【Question_id: {Question_id}】 |  \n❌ Repair stage failed, maximum retries exceeded")
                    return False
            else:
                log_msg(f"【Question_id: {Question_id}】 |  ❌ Initial return format error (attempt {attempt + 1}): actual keys are {set(statu.keys())}")

        except Exception as e:
            log_msg(f"【Question_id: {Question_id}】 |  ❌ Initial parsing failed (attempt {attempt + 1}): {e}")

    log_msg(f"【Question_id: {Question_id}】 |  \n❌ {step} stage failed, maximum retries exceeded ({max_retries})")
    return False

def GenerateSQL2(Question_id,Question, schema_json, db_name,Information_Agg, base_mess=[], db_type="sqlite", step="SQL Continuation Stage"):
    expected_keys = {
        "result_acceptable",
        "current_state",
        "sql",
        "solved_subquestions_list"
    }

    log_msg(f"\n{'-'*40}【Question_id: {Question_id}】 |  Start Stage: {step}】{'-'*40}")

    CSW = ContinueSQLWriting(Question=Question, schema_json=schema_json,Information_Agg=Information_Agg,db_type=db_type)
    CSW_mess = base_mess + [{"role": "user", "content": CSW.Prompt}]
    max_retries = 5
    log_msg(f"prompt: {CSW_mess}")
    for attempt in range(max_retries):
        try:
            log_msg(f"\n[【Question_id: {Question_id}】 |  {step}] Calling language model for the {attempt + 1} time...")

            # Modified: Added thinking parameters
            input_token_count, output_token_count, Thinking, LLM_return = LLM_output(
                messages=CSW_mess,
                model=CSW.model,
                temperature=CSW.temperature
            )
            log_msg(f"[【Question_id: {Question_id}】 |  Language Model Thinking]:\n{Thinking}")
            log_msg(f"[【Question_id: {Question_id}】 |  Language Model Output]:\n{LLM_return}")
            
            statu = extract_and_parse_json(text=LLM_return)

            logger_status.log(
                question_id=Question_id,
                step=step,
                if_in_fix="NO",
                input_token_count=input_token_count,
                output_token_count=output_token_count,
                status=statu,
                # SQL=statu.get("sql", "") # Modified: Use statu.get() safely
            )
            
            log_msg(f"[【Question_id: {Question_id}】 |  Parsing successful, returned fields]: {set(statu.keys())}")

            if set(statu.keys()) == expected_keys and statu["current_state"].lower() in {"extend", "revise", "rephrase","explore"}:
                
                # Modified: Use statu["sql"] directly
                flag,current_subsql = SQL_completion(statu["sql"], db_type)
                log_msg(f"【Question_id: {Question_id}】 |  \n✅ SQL structure is valid, starting SQL execution:\n{current_subsql}")
                status, result = db_interface(db_type=db_type, query=current_subsql, conn_info=db_name)

                if status == 0:
                    if flag==1:
                        result=result
                    else:
                        result=result+"\n*The SQL remains in an abbreviated form, but the returned answer is generated from the full version of the SQL."
                    log_msg(f"[【Question_id: {Question_id}】 |  SQL Execution Successful]\nResult:\n{result}")
                    return [
                        {"role": "assistant", "content": str(statu)},
                        {"role": "assistant", "content": "Execution result:\n" + result}
                    ],statu,current_subsql
                else:
                    log_msg(f"【Question_id: {Question_id}】 |  \n⚠️ Initial SQL execution failed:\nError Message:\n{result}")
                    log_msg(f"{'-'*20}【【Question_id: {Question_id}】 |  Entering Repair Process】{'-'*20}")


                    fix_mess = CSW_mess + [
                        {"role": "assistant", "content": str(statu)},
                        {"role": "user", "content": result + "\nPlease analyze and fix the current " + db_type + " SQL error" +
                        (", and in certain cases, some columns in the current database do not support the use of double quotes. Please detect and handle such situations" if db_type == "snow" else "") +
                        (", and pay close attention to BigQuery's quoting rules: use backticks (`` ` ``) for identifiers (like `project.dataset.table`) and single/double quotes (' or \") for string values. Misusing quotes is a very common error." if db_type == "bigquery" else "") +
                        ", and return the corrected SQL in the same Markdown JSON format (including the key names) with ```json```."}
                    ]

                    log_msg(f"fix prompt: {fix_mess}")
                    for fix_attempt in range(max_retries):
                        try:
                            log_msg(f"\n[【Question_id: {Question_id}】 |  Repair Attempt #{fix_attempt + 1}] Calling language model for repair...")
                            
                            # Modified: Added thinking parameters
                            input_token_count, output_token_count, Thinking, fix_return = LLM_output(
                                messages=fix_mess,
                                model=CSW.model,
                                temperature=CSW.temperature
                            )
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair LLM Thinking]:\n{Thinking}")
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair LLM Output]:\n{fix_return}")
                            
                            fix_statu = extract_and_parse_json(fix_return)

                            logger_status.log(
                                question_id=Question_id,
                                step=f"{step} Repair",
                                if_in_fix="YES",
                                input_token_count=input_token_count,
                                output_token_count=output_token_count,
                                status=fix_statu,
                                # SQL=fix_statu.get("sql", "")
                            )
                            
                            log_msg(f"[【Question_id: {Question_id}】 |  Repair parsing successful, returned fields]: {set(fix_statu.keys())}")

                            if set(fix_statu.keys()) == expected_keys and fix_statu["current_state"].lower() in {"extend", "revise", "rephrase","explore"}:
                                flag,current_subsql = SQL_completion(fix_statu["sql"], db_type)
                                log_msg(f"[【Question_id: {Question_id}】 |  Attempting to execute repaired SQL]:\n{current_subsql}")
                                status, result = db_interface(db_type=db_type, query=current_subsql, conn_info=db_name)

                                if status == 0:
                                    log_msg(f"[【Question_id: {Question_id}】 |  Repaired SQL Execution Successful]\nResult:\n{result}")
                                    if flag==1:
                                        result=result
                                    else:
                                        result=result+"\n*The SQL remains in an abbreviated form, but the returned answer is generated from the full version of the SQL."
                                    return [
                                        {"role": "user", "content": str(fix_statu)},
                                        {"role": "assistant", "content": "Execution result:\n" + result}
                                    ],fix_statu,current_subsql
                                else:
                                    log_msg(f"【Question_id: {Question_id}】 |  ⚠️ Repair attempt {fix_attempt + 1} failed:\n{result}")

                                    fix_mess += [
                                        {"role": "user", "content": str(fix_statu)},
                                        {"role": "user", "content": result + "\nPlease analyze and fix the current " + db_type + " SQL error again" +
                                        (", and in certain cases, some columns in the current database do not support the use of double quotes. Please detect and handle such situations" if db_type == "snow" else "") +
                                        (", and pay close attention to BigQuery's quoting rules: use backticks (`` ` ``) for identifiers (like `project.dataset.table`) and single/double quotes (' or \") for string values. Misusing quotes is a very common error." if db_type == "bigquery" else "") +
                                        ", and return the corrected SQL in the same Markdown JSON format (including the key names) with ```json```."}
                                    ]
                            else:
                                log_msg(f"【Question_id: {Question_id}】 |  ❌ Repair attempt {fix_attempt + 1} returned incorrect format: actual keys are {set(fix_statu.keys())}")

                        except Exception as e:
                            log_msg(f"【Question_id: {Question_id}】 |  ❌ Repair attempt {fix_attempt + 1} parsing failed: {e}")

                    log_msg(f"【Question_id: {Question_id}】 |  \n❌ Repair stage failed, maximum retries exceeded")
                    return False
            else:
                log_msg(f"【Question_id: {Question_id}】 |  ❌ Initial return format error (attempt {attempt + 1}): actual keys are {set(statu.keys())}")

        except Exception as e:
            log_msg(f"【Question_id: {Question_id}】 |  ❌ Initial parsing failed (attempt {attempt + 1}): {e}")

    log_msg(f"【Question_id: {Question_id}】 |  \n❌ {step} stage failed, maximum retries exceeded ({max_retries})")
    return False

def GenerateSQL(Question_id, Question, Col, schema_json, db_name,Information_Agg, base_mess=[], db_type="sqlite", max_total_steps=20):
    log_msg(f"【Question_id: {Question_id}】 |  Starting SQL Generation Pipeline. Max steps: {max_total_steps}")
    step_counter = 0
    pkl_filename = f"{Question_id}_IntermediateSQL.pkl"
    latest_sql = None
    final_status = None
    
    temp_sql = None 

    base_mess = base_mess.copy()
    initial_base_mess = base_mess.copy()
    latest_mess = []

    log_msg(f"【Question_id: {Question_id}】 |  Attempting to load intermediate progress from {pkl_filename}")
    try:
        loaded_data = save_or_load_pickle(filename=pkl_filename, mode='load')
        log_msg(f"【Question_id: {Question_id}】 |  ✅ Successfully loaded data from {pkl_filename}. Skipping Stage 1 & 2.")
        initial_base_mess = loaded_data['initial_base_mess']
        latest_mess = loaded_data['latest_mess']
        final_status = loaded_data.get('final_status')
        latest_sql = loaded_data.get('latest_sql')
        temp_sql = loaded_data.get('temp_sql') 
        step_counter = loaded_data.get('step_counter', 0)
        log_msg(f"【Question_id: {Question_id}】 |  Loaded state: step_counter={step_counter}, latest_sql=\n{latest_sql}")

    except FileNotFoundError:
        log_msg(f"【Question_id: {Question_id}】 |  No intermediate file found. Starting from Stage 1.")
        log_msg(f"【Question_id: {Question_id}】 |  --- Entering Stage 1: Initial SQL Generation --- (Step {step_counter + 1})")
        
        step1_result = GenerateSQL1(Question_id=Question_id,Question=Question, schema_json=schema_json, Information_Agg=Information_Agg,db_name=db_name, base_mess=base_mess,db_type=db_type)
        step_counter += 1

        # --- CHANGE 1 START ---
        if not step1_result:
            log_msg(f"【Question_id: {Question_id}】 |  ❌ Stage One failed (returned None). Terminating.")
            return {"temp_SQL": None, "final_SQL": None}, step_counter

        latest_mess, statu, current_subsql = step1_result
        
        if not current_subsql:
            log_msg(f"【Question_id: {Question_id}】 |  ❌ Stage One failed (no SQL generated). Terminating.")
            return {"temp_SQL": None, "final_SQL": None}, step_counter
        # --- CHANGE 1 END ---

        base_mess.extend(latest_mess)
        # --- CHANGE 2---
        latest_sql = current_subsql
        final_status = statu
        log_msg(f"【Question_id: {Question_id}】 |  ✅ Stage One successful. Intermediate SQL:\n{latest_sql}")

        log_msg(f"【Question_id: {Question_id}】 |  --- Entering Stage 2: SQL Continuation Loop ---")
        while step_counter < max_total_steps:
            log_msg(f"【Question_id: {Question_id}】 |  --- Stage 2 Iteration (Step {step_counter + 1}) ---")
            
            # 调用 GenerateSQL2
            step2_result = GenerateSQL2(
                Question_id=Question_id,
                Question=Question,
                schema_json=schema_json,
                db_name=db_name,
                Information_Agg=Information_Agg,
                base_mess=base_mess,
                db_type=db_type
            )
            step_counter += 1

            # --- CHANGE 3 START ---
            if not step2_result:
                log_msg(f"【Question_id: {Question_id}】 |  ⚠️ Stage Two interrupted due to failure (returned None). Returning last valid SQL.")
                return {"temp_SQL": latest_sql, "final_SQL": latest_sql}, step_counter
            
            latest_mess, statu, temp_sql_from_step2 = step2_result
            
            if not temp_sql_from_step2:
                log_msg(f"【Question_id: {Question_id}】 |  ⚠️ Stage Two interrupted due to failure (no SQL generated). Returning last valid SQL.")
                return {"temp_SQL": latest_sql, "final_SQL": latest_sql}, step_counter
            # --- CHANGE 3 END ---
            
            base_mess.extend(latest_mess)
            # --- CHANGE 4---
            latest_sql = temp_sql_from_step2
            temp_sql = temp_sql_from_step2 
            final_status = statu
            log_msg(f"【Question_id: {Question_id}】 |  ✅ Stage Two iteration successful. Current SQL:\n{latest_sql}")

            if statu.get("result_acceptable") and statu.get("current_state", "").lower() == "rephrase":
                log_msg(f"【Question_id: {Question_id}】 |  ✅ Stage Two termination condition met (state='rephrase'). Saving progress to {pkl_filename}.")
                save_or_load_pickle(data={
                    "initial_base_mess": initial_base_mess,
                    "latest_mess": latest_mess,
                    "final_status": final_status,
                    "latest_sql": latest_sql,
                    "temp_sql": temp_sql,
                    "step_counter": step_counter,
                }, filename=pkl_filename, mode='save')
                break
        else:
            log_msg(f"【Question_id: {Question_id}】 |  ❌ Maximum steps exceeded in Stage Two. Returning last valid SQL.")
            return {"temp_SQL": temp_sql, "final_SQL": latest_sql}, step_counter
        
    return {"temp_SQL": temp_sql, "final_SQL": latest_sql}, step_counter

#---- Generation-State Evolution----

def workflow(Question_id, Question, schema_json, db_name,db_type="sqlite"):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg(f"\n\n\n------------------------------------datetime：{current_time}------------------------------------")
    log_msg(f"\n\n-----------------Starting workflow for question: {Question_id}-----------------\n")
    log_msg(f"Question: {Question}")
    
    # Set cache filenames
    base_pickle_filename = f"{Question_id}_DS.pkl"
    infor_ag_filename=f"{Question_id}_IA_DS.pkl"
    
    # Initial System Prompt
    base_messages = []

    # [Stage] Database Exploration
    log_msg("\n--- Starting Stage: Database Exploration ---")
    try:
        query_list_2 = save_or_load_pickle(filename=base_pickle_filename, mode='load')
        log_msg(f"✅ Cached DB exploration results loaded, skipping stage: {base_pickle_filename}")
    except FileNotFoundError:
        log_msg(f"⚠️ Cache not found, executing live database exploration and saving to: {base_pickle_filename}")        
        # Fine-grained exploration
        query_list_2 = Fine_grained_Exploration_func(Question_id=Question_id,Question=Question, schema_json=schema_json, db_name=db_name, base_mess=base_messages,db_type=db_type)
        # Save message sequence after exploration
        save_or_load_pickle(data=query_list_2, filename=base_pickle_filename, mode='save')
        log_msg("✅ Database exploration results saved.")

    # [Stage] Information Aggregation
    log_msg("\n--- Starting Stage: Information Aggregation ---")
    try:
        infor_ag = save_or_load_pickle(filename=infor_ag_filename, mode='load')
        log_msg(f"✅ Cached Information Aggregation loaded, skipping stage: {infor_ag_filename}")
    except FileNotFoundError:
        log_msg(f"⚠️ Cache not found, executing live information aggregation and saving to: {infor_ag_filename}")
        infor_ag = Information_Summary(Question_id=Question_id,Question=Question,schema_json=schema_json,DB_Exploration=query_list_2)
        save_or_load_pickle(data=infor_ag, filename=infor_ag_filename, mode='save')
        log_msg("✅ Information Aggregation results saved.")
        
    # [Stage] Main SQL Generation
    log_msg("\n--- Starting Stage: Main SQL Generation Pipeline ---")

    Finished_SQL,step_counter = GenerateSQL(Question_id=Question_id,Question=Question,Col="", schema_json=schema_json, db_name=db_name, Information_Agg=infor_ag,base_mess=base_messages,db_type=db_type)
    
    log_msg("\n--- Workflow Finished ---")
    log_msg(f"Total steps in generation pipeline: {step_counter}")
    log_msg(f"Final SQL Result:\n{Finished_SQL}")
    return Finished_SQL,step_counter

def get_instance_ids(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['instance_id'] for item in data]

def log_msg(msg):
    logger.info(msg)

RESULTS_LOCK = Lock()

def process_entry(entry,MAX_MSchema_TOKEN):
    """
    Process a single task entry, including log context setup, RAG input construction,
    SQL inference call, and exception handling.
    """
    init_time = time.time()
    question_id = entry['instance_id']
    db_id = entry.get('db_id') or entry.get('db')
    db_type = detect_db_type(question_id)
    
    # --- Set log context (helps identify the source of log messages) ---
    log_context.question_id = question_id
    log_context.db_id = db_id

    try:
        log_msg(f"🚀 Starting task: instance_id={question_id}, db_id={db_id}")

        # Get instruction and evidence
        question = entry.get('instruction') or entry.get('question')
        evidence = entry.get('evidence') or ""
        SL = entry["table"]

        # Construct user input (with optional evidence)
        if evidence:
            user_input = f"[Evidence]\n{evidence}\n[Question]\n{question}\n"
        else:
            user_input = f"[Question]\n{question}\n"

        log_msg("📨 Input constructed, invoking workflow...")
        ## When the context exceeds a certain limit, use DDL statements directly.
        # TODO: A hierarchical pruning approach can be adopted to maximize the score: https://github.com/Snowflake-Labs/ReFoRCE/blob/o3/methods/ReFoRCE/reconstruct_data.py
        if get_token_count(M_Schema(SL=SL, db_id=db_id, db_type=db_type))>MAX_MSchema_TOKEN:
            schema_json=generate_ddl_from_json(db_id=db_id,table_list=SL,db_type=db_type)
        else:
            schema_json=M_Schema(SL=SL, db_id=db_id, db_type=db_type)
        # Execute core logic (SQL inference)
        Pre_SQL, step_counter = workflow(
            Question_id=question_id,
            Question=user_input,
            schema_json=schema_json,
            db_name=db_id,
            db_type=db_type
        )

        log_msg(f"✅ Task completed successfully. (Steps: {step_counter})")

        # Under the current implementation, temp_SQL and final_SQL are completely identical.
        entry["temp_SQL"] = Pre_SQL["temp_SQL"]
        entry["final_SQL"] = Pre_SQL["final_SQL"]
        entry["Step_counter"] = step_counter

        end_time= time.time()
        time_cost = end_time - init_time

        logger_status.log(
                question_id=question_id,
                step="Time Cost",
                if_in_fix="NO",
                input_token_count=init_time,
                output_token_count=end_time,
                status=time_cost  
            )

        return entry

    except Exception as e:
        logger.error(f"❌ Exception occurred while processing task {question_id}: {e}", exc_info=True)
        return None

    finally:
        # --- Clean up log context to prevent thread contamination ---
        for attr in ['question_id', 'db_id']:
            if hasattr(log_context, attr):
                delattr(log_context, attr)

def save_result_safely(result, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with RESULTS_LOCK:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try: all_results = json.load(f)
                except json.JSONDecodeError: all_results = []
        else:
            all_results = []
        all_results.append(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        log_msg(f"Result saved safely. Current result count: {len(all_results)}")



if __name__ == "__main__":
    # --- 0. Startup Warning & Delay ---
    print("\n" + "="*60)
    print('''
    Using DeepSeek directly execution takes a very long time. You can refer to the author of Spider2.0 and manually use multiple terminals to run!
    list_1, list_2,list_3,list_4,list_5 =[all_list[i::5] for i in range(5)]
    ''')
    print("="*60)
    
    print("\nWaiting for 15 seconds before starting...")
    time.sleep(15)

    # --- 1. Argument Parsing (For Shell & Python convenience) ---
    parser = argparse.ArgumentParser(description="Spider2-Lite Runner")
    
    # Input Path (Mandatory)
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to the input JSON file (e.g., data_lite/spider2-lite.json)"
    )
    
    # Data Sub Directory (Optional, defaults to Result_MonthDayHourMinute)
    default_sub_dir = datetime.now().strftime("Result_%m%d%H%M")
    parser.add_argument(
        "--data_sub_dir", 
        type=str, 
        default=default_sub_dir, 
        help=f"Subdirectory for results (Default: {default_sub_dir})"
    )
    
    # Multi-path Toggle (Optional, defaults to False)
    parser.add_argument(
        "--multi_path", 
        action="store_true", 
        help="Enable multi-path execution (Run 1-5 times). Default is 1 time."
    )

    args = parser.parse_args()

    # --- 2. Configuration & Path Management ---
    
    # ROOT_DIR: Automatically set to the directory containing this script
    ROOT_DIR = Path(__file__).resolve().parent
    
    # Input Path handling (Convert to Path object)
    # If the user provides a relative path, it joins with ROOT_DIR; if absolute, it uses it directly.
    INPUT_PATH = Path(args.input_path)
    if not INPUT_PATH.is_absolute():
        INPUT_PATH = ROOT_DIR / INPUT_PATH

    # Data Storage Directory
    DATA_SUB_DIR = args.data_sub_dir
    WORK_DIR = ROOT_DIR / DATA_SUB_DIR
    
    # Execution Flags
    IF_MULTI_PATH = args.multi_path
    MAX_MSCHEMA_TOKEN = 55535
    WAIT_MINUTES_BEFORE_EXIT = 0
    
    # Database IDs to exclude
    EXCLUDE_IDS = {"bq109"} # "bq064", "bq352", "bq445", "sf_bq372"

    # Ensure directories exist
    if not WORK_DIR.exists():
        os.makedirs(WORK_DIR, exist_ok=True)

    # --- 3. Get Task List ---
    # Note: get_instance_ids must be defined in your imports
    all_list = [
        x for x in get_instance_ids(json_path=str(INPUT_PATH)) 
        if x not in EXCLUDE_IDS
    ]

    # --- 4. Main Loop ---
    # Determine loop range based on IF_MULTI_PATH
    run_range = range(1, 6) if IF_MULTI_PATH else range(1, 2)

    for sql_item in all_list:
        for run_id in run_range:
            
            run_key = f"{sql_item}_{run_id}"

            # Construct related file paths (using pathlib for automatic separator handling)
            outcome_path = WORK_DIR / "outcome" / f"{run_key}_result.json"
            log_file_path = WORK_DIR / "log" / run_key / f"main_{run_key}.log"
            status_file_path = WORK_DIR / "log" / run_key / f"status_{run_key}.jsonl"
            temp_path = WORK_DIR / "temp" / run_key
            
            # Ensure sub-directories exist
            os.makedirs(outcome_path.parent, exist_ok=True)
            os.makedirs(log_file_path.parent, exist_ok=True)
            os.makedirs(temp_path, exist_ok=True)

            # Logger Initialization (Convert to str for compatibility)
            # Note: JsonLogger and setup_logger must be defined in your imports
            logger_status = JsonLogger(log_file_path=str(status_file_path))
            logger = setup_logger(str(log_file_path), logger_name=f"logger_for_{run_key}")
            
            # Convert paths to strings for functions that don't support Path objects
            output_path_str = str(outcome_path)
            input_path_str = str(INPUT_PATH)

            log_msg("=========================================================")
            log_msg(f"=== Starting Spider2.0-Lite for Item: {sql_item} | Run: {run_id} ===")
            log_msg("=========================================================")

            waited_once = False
            
            while True:
                # Check if input file exists
                if not os.path.exists(input_path_str):
                    log_msg(f"Input file {input_path_str} not found. Waiting...")
                    time.sleep(60)
                    continue

                # Read all task data
                with open(input_path_str, 'r', encoding='utf-8') as f:
                    all_tasks_data = json.load(f)

                # Read processed results (Resume logic)
                processed_ids = set()
                if os.path.exists(output_path_str):
                    with open(output_path_str, 'r', encoding='utf-8') as fout:
                        try:
                            result_data = json.load(fout)
                            # Ensure result_data is a list
                            if isinstance(result_data, list):
                                processed_ids = {entry.get('instance_id') for entry in result_data}
                        except json.JSONDecodeError:
                            pass # result_data is empty or invalid

                # Filter tasks for current sql_item
                entries_to_process = [
                    entry for entry in all_tasks_data
                    if entry['instance_id'] == sql_item and entry['instance_id'] not in processed_ids
                ]

                # If no tasks to process
                if not entries_to_process:
                    if waited_once:
                        log_msg(f"Waited for {WAIT_MINUTES_BEFORE_EXIT} minutes and still no new tasks for {run_key}.")
                        break
                    else:
                        log_msg(f"No new tasks for {run_key}. Waiting {WAIT_MINUTES_BEFORE_EXIT} minutes...")
                        waited_once = True
                        if WAIT_MINUTES_BEFORE_EXIT > 0:
                            time.sleep(WAIT_MINUTES_BEFORE_EXIT * 60)
                        continue

                waited_once = False
                
                # Execute processing
                for entry in entries_to_process:
                    question_id = entry['instance_id']
                    try:
                        # process_entry must be defined
                        result = process_entry(entry, MAX_MSCHEMA_TOKEN)
                        
                        if result:
                            save_result_safely(result, output_path_str)
                        else:
                            log_msg(f"[{question_id}] ⚠️ Null result returned.")
                    except Exception as e:
                        log_msg(f"[{question_id}] ❌ Exception: {e}")
