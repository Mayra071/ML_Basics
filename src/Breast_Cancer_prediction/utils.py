import os
import sys
from src.Breast_Cancer_prediction.exception import CustomException
from src.Breast_Cancer_prediction.logger import logging
import pandas as pd
import numpy as np
import pickle


from dotenv import load_dotenv
# Generic functionality for data reading


# read data from from my local system
load_dotenv()

host = os.getenv("HOST")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")

# function to read data
def read_data():
    
    """Reads data from database.

    """
    logging.info("Reading data from database")
    try:
        # mydb=pymysql.connect(
        # host=host,
        # user=user,
        # password=password,
        # database=database
        # )
        
        # logging.info("Successfully connected to the database")
        # df=pd.read_sql('select * from breast_cancer_data', con=mydb)
        
        file_path = r'C:\Machine Learning\New folder\Dataset_1\Breast_Cancer.csv'
        df = pd.read_csv(file_path)
        logging.info(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        raise CustomException(e, sys)

# function to save object
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)