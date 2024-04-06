import logging
import os


def configure_logging(pipeline_dir):
    """
    Configure the root logger to write logs to a file in the specified pipeline directory.

    Args:
        pipeline_dir (str): The directory for the pipeline execution.
    """
    path_to_logs = os.path.expanduser(pipeline_dir)
    os.makedirs(path_to_logs, exist_ok=True)
    log_file = os.path.join(path_to_logs, "pipeline.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, stream_handler],
    )


def init_logger(name, level=logging.DEBUG):
    """
    Set up a logger with the specified name and logging level.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
