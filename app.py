from src.Breast_Cancer_prediction.logger import logging
from src.Breast_Cancer_prediction.exception import CustomException
from dotenv import load_dotenv
import sys
from src.Breast_Cancer_prediction.components.data_ingestion import DataIngestion, DataIngestionConfig


if __name__ == "__main__":
    logging.info("Application started")
    # Your application code here
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        # logging.info("Data ingestion completed successfully")
    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e, sys)