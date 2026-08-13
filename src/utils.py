import logging

def setup_logger(name, log_file=None, level=logging.INFO, queue=None):
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

    # support for multiprocesing logging
    if queue:
        logger.addHandler(queue)

    return logger

def worker_process(queue, worker_id):
    """
    Worker function executed in child processes.
    """
    # 1. Configure worker to route logs exclusively to the shared queue
    queue_handler = logging.handlers.QueueHandler(queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    # 2. Log exactly as normal
    logging.info(f"Worker {worker_id} is starting intensive computation...")
    time.sleep(1)
    logging.warning(f"Worker {worker_id} completed successfully.")
