from src.Breast_Cancer_prediction.logger import logging
from src.Breast_Cancer_prediction.exception import CustomException
from dotenv import load_dotenv
import sys
from src.Breast_Cancer_prediction.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.Breast_Cancer_prediction.components.data_transformation import DataTransformation, DataTransformationConfig
# from src.Breast_Cancer_prediction.components.model_trainer

if __name__ == "__main__":
    logging.info("Application started")
    # Your application code here
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")

        # Data Transformation
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation( train_data_path, test_data_path)
        logging.info("Data transformation completed successfully")
        
    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e, sys)