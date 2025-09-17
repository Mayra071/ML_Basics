import logging
import os
from datetime import datetime

# Creating logs directory
LOG_FILE = f"breast_cancer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Define the path for the logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Setting up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)