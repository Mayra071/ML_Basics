import os
import sys
from src.Breast_Cancer_prediction.exception import CustomException
from src.Breast_Cancer_prediction.logger import logging
import pandas as pd
import numpy as np
from src.Breast_Cancer_prediction.utils import read_data

from sklearn.model_selection import train_test_split



from dataclasses import dataclass


# Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingetstion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        
        logging.info("Entered the data ingestion method or component")
        try:
            # read the dataset
            # We can read the dataset from a local path or a database like MySQL, MongoDB, PostgreSQL, etc.
            
            df=read_data()
            
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingetstion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingetstion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            # train data path
            os.makedirs(os.path.dirname(self.ingetstion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingetstion_config.train_data_path, index=False, header=True)
            
            # test data path
            os.makedirs(os.path.dirname(self.ingetstion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.ingetstion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            return (
                self.ingetstion_config.train_data_path,
                self.ingetstion_config.test_data_path
            )
            
            
        except Exception as ex:
            raise CustomException(ex, sys)