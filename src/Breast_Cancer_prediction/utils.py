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
    """
    Run grid search for each model and return classification reports + best models.
    Returns:
        reports (dict): { model_name: { "report_text": str, "report_dict": dict, "best_params": dict } }
        best_models (dict): { model_name: fitted_estimator }
    """
    reports = {}
    best_models = {}
    
    try:
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            logging.info(f"Grid searching {model_name} with params: {param_grid}")
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(X_test)

            report_text = classification_report(y_test, y_test_pred)
            report_dict = classification_report(y_test, y_test_pred, output_dict=True)

            logging.info(f"{model_name} best params: {gs.best_params_}")
            logging.info(f"{model_name} classification report:\n{report_text}")

            reports[model_name] = {
                "report_text": report_text,
                "report_dict": report_dict,
                "best_params": gs.best_params_
            }
            best_models[model_name] = best_model

        return reports, best_models
    except Exception as e:
        raise CustomException(e, sys)
