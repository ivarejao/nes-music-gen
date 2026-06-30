import logging

def setup_logger(name, log_file=None, level=logging.INFO):
    """Sets up a logger that writes messages to a file."""
    
    # 1. Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(name)s [%(levelname)s] %(message)s')
    # 2. Create console handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger