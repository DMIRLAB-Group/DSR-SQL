import logging
import threading
import json,os,sys
from datetime import datetime
from typing import Dict, Optional, Literal

# --- 1. Thread-local storage for passing context information ---
log_context = threading.local()

# --- 2. Custom JSON Formatter ---
class JsonFormatter(logging.Formatter):
    """
    Formats log records into a JSON string (JSON Lines format), ensuring field order:
    timestamp -> question_id -> level -> message -> thread_name
    """
    def format(self, record):
        json_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M"),
            "question_id": getattr(log_context, 'question_id', 'N/A'),
            "level": record.levelname,
            "message": record.getMessage(),
            "thread_name": record.threadName,
        }
        return json.dumps(json_record, ensure_ascii=False)


# --- 3. Context Filter for console output ---
class ContextFilter(logging.Filter):
    """
    This filter injects information from the thread-local context into the log record,
    so that the formatter can access it.
    """
    def filter(self, record):
        record.question_id = getattr(log_context, 'question_id', 'N/A')
        return True

# --- Modified Function ---
def setup_logger(log_file_path, logger_name):
    """
    Configures and returns a logger instance with the specified name.
    Each logger_name will only be configured once.

    :param log_file_path: The output path for the log file.
    :param logger_name: The unique name for the logger.
    """
    logger = logging.getLogger(logger_name)

    # If this specific logger has already been configured with handlers, return it directly to avoid duplicate handlers.
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    # Set propagate to False to prevent log messages from propagating up to the root logger, avoiding duplicate output.
    logger.propagate = False

    # --- Ensure the log directory exists ---
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            # Handle potential race condition when creating directory concurrently.
            if not os.path.isdir(log_dir):
                raise

    # --- File Handler ---
    # Writes logs to the specified file.
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    # Assuming you have a custom JsonFormatter
    # file_handler.setFormatter(JsonFormatter())
    # If not, you can use the standard Formatter
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # --- Console Handler ---
    # Outputs logs to standard output (e.g., the terminal).
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    human_readable_formatter = logging.Formatter(
        '[%(asctime)s] [Logger: %(name)s] [Q_ID: %(question_id)s] [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Add a filter to ensure 'question_id' exists
    stream_handler.addFilter(ContextFilter())
    stream_handler.setFormatter(human_readable_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class JsonLogger:
    """
    A class for logging in JSON format (JSON Lines).

    Specify the log file path during initialization, and then call the log method
    to record logs without repeatedly passing the file path.
    """

    def __init__(self, log_file_path: str):
        """
        Initializes the Logger.

        Args:
            log_file_path (str): The full path to the log file.
                                 If the directory containing the file does not exist, it will be created automatically.
        """
        self.log_file_path = log_file_path
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Ensures that the directory for the log file exists."""
        try:
            # Get the directory path of the log file
            log_dir = os.path.dirname(self.log_file_path)
            # If the directory path is not empty (i.e., the file is not in the current directory), create the directory
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Log directory created: {log_dir}")
        except OSError as e:
            print(f"Error: Failed to create log directory '{log_dir}'.")
            print(f"Error details: {e}")
            # Raise an exception because subsequent writes will fail if the directory cannot be created
            raise

    def log(
        self,
        question_id: str,
        step: int,
        if_in_fix: Literal["YES", "NO"],
        input_token_count: int,
        output_token_count: int,
        status: Optional[Dict] = None,
        SQL: Optional[str]=None
    ) -> None:
        """
        Records a log entry in JSON format.

        Args:
            question_id (str): The question ID.
            step (int): The current step number.
            if_in_fix (Literal["YES", "NO"]): Whether it is in the fix process.
            input_token_count (int): The number of input tokens.
            output_token_count (int): The number of output tokens.
            status (Optional[Dict]): A dictionary describing the status, can be None.
            SQL (Optional[str]): The SQL query string, can be None.
        """
        try:
            # 1. Create the log data dictionary
            log_data = {
                "TIME": datetime.now().isoformat(),
                "Question_id": question_id,
                "step": step,
                "if_in_fix": if_in_fix,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "status": status,
                "SQL": SQL
            }

            # 2. Convert the dictionary to a JSON string
            json_string = json.dumps(log_data, ensure_ascii=False)

            # 3. Open the file in append mode and write the log
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json_string + '\n')

        except (IOError, TypeError) as e:
            print(f"Error: Failed to write to log file '{self.log_file_path}'.")
            print(f"Error details: {e}")