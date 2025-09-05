from src.Breast_Cancer_prediction.logger import logging
from src.Breast_Cancer_prediction.exception import CustomException
import sys


if __name__ == "__main__":
    logging.info("Application started")
    # Your application code here
    try:
        a=1/1  # Example to raise an exception
    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e, sys)