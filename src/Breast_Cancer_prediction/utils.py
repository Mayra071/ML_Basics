import os
import sys
from src.Breast_Cancer_prediction.exception import CustomException
from src.Breast_Cancer_prediction.logger import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

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
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            train_model_score = accuracy_score(y_train, y_train_pred)
            logging.info(f'train_model_score: {train_model_score}')
            
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            logging.info(f'test_model_score: {test_model_score}')
            report[list(models.keys())[i]] =classification_report(y_test, y_test_pred)
            
        return report
    except Exception as e:
        raise CustomException(e, sys)