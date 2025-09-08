import os
import sys  
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
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
                    'solver': ['liblinear'],
                    'max_iter': [1000],
                    'class_weight': ['balanced']
                },
                "XGBoost": {
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'max_depth': [4, 6],
                    'objective': ['binary:logistic'],
                    'eval_metric': ['logloss'],
                    'n_jobs': [-1],
                    'tree_method': ['hist'],
                    'n_estimators': [150]
                }
            }
            
            # get classification reports for all models
            reports = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # Also save the best model by simple accuracy parsing from reports (optional)
            # Compute accuracy from predictions again for saving best model
            # # Re-run grid search quickly to get best estimator for each, then pick highest accuracy
           
            # accuracies = {}
            # best_estimators = {}
            # for model_name, model in models.items():
            #     gs = GridSearchCV(model, params.get(model_name, {}), cv=3, n_jobs=-1, scoring='accuracy', verbose=0)
            #     gs.fit(X_train, y_train)
            #     best_model = gs.best_estimator_
            #     acc = accuracy_score(y_test, best_model.predict(X_test))
            #     accuracies[model_name] = acc
            #     best_estimators[model_name] = best_model
            # best_model_name = max(accuracies, key=accuracies.get)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                # obj=best_estimators[best_model_name]
                obj=reports
            )
            logging.info("Best model saved")
            
            return reports
            
        except Exception as e:
            raise CustomException(e, sys)
            logging.info("Exception occurred during model training")
        
        logging.info("Model training completed")