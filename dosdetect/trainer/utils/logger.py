import logging
import os

from ..config import Config

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with the specified name, log file, log directory, and logging level.

    Args:
        name (str): The name of the logger.
        log_file (str): The name of the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Expand the user's home directory if necessary, safe to use for all paths
    expanded_log_dir = os.path.expanduser(Config.LOG_DIR)
    try:
        os.makedirs(expanded_log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")  # Catch and print any error

    log_path = os.path.join(expanded_log_dir, log_file)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger