import sys
import os
import logging
from contextlib import redirect_stdout


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def configure_logging(pipeline_dir):
    """
    Configure the root logger to write logs to a file in the specified pipeline directory
    and redirect stdout to the logger.

    Args:
        pipeline_dir (str): The directory for the pipeline execution.
    """
    path_to_logs = os.path.expanduser(pipeline_dir)
    os.makedirs(path_to_logs, exist_ok=True)
    log_file = os.path.join(path_to_logs, "pipeline.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    stdout_logger = logging.getLogger("STDOUT")
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("STDERR")
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl


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
