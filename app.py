from src.Breast_Cancer_prediction.logger import logging
from src.Breast_Cancer_prediction.exception import CustomException
from dotenv import load_dotenv
import sys
from src.Breast_Cancer_prediction.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.Breast_Cancer_prediction.components.data_transformation import DataTransformation, DataTransformationConfig
from src.Breast_Cancer_prediction.components.model_trainer import ModelTrainer, ModelTrainerConfig

import os
import pandas as pd
import numpy as np

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

        # Initiate data transformation
        transformed_train_data,transformed_test_data,_=data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Logging transformed data
        logging.info("Data transformation completed successfully")
        # print transformed data
        logging.info(f"Transformed Train Data:\n{transformed_train_data[:5]}")
        logging.info(f"Transformed Test Data:\n{transformed_test_data[:5]}")

        # Model Trainer
        model_trainer = ModelTrainer()
        name,acc=model_trainer.initiate_model_trainer(transformed_train_data, transformed_test_data)
        logging.info("Model training completed successfully")
        logging.info(f"Best Model: {name} with accuracy: {acc}")

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e, sys)