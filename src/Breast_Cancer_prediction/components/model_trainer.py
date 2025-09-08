import os
import sys  
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.Breast_Cancer_prediction.logger import logging
from  src.Breast_Cancer_prediction.exception import CustomException
from src.Breast_Cancer_prediction.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            logging.info("Training the model")
            
            models={
                "Logistic Regression": LogisticRegression(),
                "XGBoost": xgb.XGBClassifier()
            }
            
            params={
                "Logistic Regression": {
                    'C': [0.1, 1.0, 3.0],
                    'penalty': ['l1', 'l2'],
                    'max_iter': [1000],
                    'random_state': [42],
                    'class_weight': ['balanced', None],
                    'solver': ['liblinear', 'saga']
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'max_depth': [4, 6, 8],
                    'eval_metric': ['logloss'],
                    'feature_selector':['shuffle'],
                    'use_label_encoder': [False],
                    'n_estimators': [50, 100, 200]
                    
                }
            }
            
            # model evaluation
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # to get the best model score from the dictionary
            best_model_score = max(sorted(model_report['accuracy'].values()))
            
            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")
            # saving the object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )
            logging.info("Best model saved")
            
            return best_model_name, best_model_score
            
            
        except Exception as e:
            raise CustomException(e, sys)
            logging.info("Exception occurred during model training")
        
        logging.info("Model training completed")