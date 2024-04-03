class Config:
    LOG_DIR = '~/.dosdetect/logs'
    MODEL_DIR = '~/.dosdetect/models'

    @classmethod
    def set_log_dir(cls, log_dir):
        cls.LOG_DIR = log_dir or cls.LOG_DIR

    @classmethod
    def set_model_dir(cls, model_dir):
        cls.MODEL_DIR = model_dir or cls.MODEL_DIR