import logging
import os
from datetime import datetime

#make log directory , if already exists : ignore
log_dir = os.path.join(os.getcwd(),'logs')
os.makedirs(log_dir , exist_ok=True)

#file name
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#log file path
log_file_path = os.path.join(log_dir , LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.DEBUG
)

