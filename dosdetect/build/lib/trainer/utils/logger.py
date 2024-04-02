import logging

LOG_DIR = 'logs'

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with the specified name, log file, and logging level.

    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f'{LOG_DIR}/{log_file}')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger