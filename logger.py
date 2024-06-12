import logging
import os
from datetime import datetime

LOGS_FOLDER = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%H_%M_%S')}.log"
LOG_FORMAT = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

os.makedirs(LOGS_FOLDER, exist_ok=True)
log_file_path = os.path.join(LOGS_FOLDER, LOG_FILE)

logging.basicConfig(filename=log_file_path, format=LOG_FORMAT, level=logging.INFO)