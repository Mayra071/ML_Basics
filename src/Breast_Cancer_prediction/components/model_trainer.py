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
from src.Breast_Cancer_prediction.mlflow_utils import (
    set_tracking_uri,
    set_experiment,
    start_run,
    log_params,
    log_metrics,
    log_sklearn_model,
    log_xgboost_model,
    log_artifact,
    log_pandas_dataset,
)
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def _flatten_report(self, report_dict):
        flat = {}
        for label, section in report_dict.items():
            if isinstance(section, dict):
                for metric_name, value in section.items():
                    if isinstance(value, (int, float)):
                        flat[f"{label}_{metric_name}"] = float(value)
            elif isinstance(section, (int, float)):
                flat[label] = float(section)
        return flat
        
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
            
            # evaluate and get best models + reports
            reports, best_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # Configure MLflow (env overrides supported)
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            experiment_name = os.getenv("MLFLOW_EXPERIMENT", "Breast_Cancer_Prediction")
            set_tracking_uri(tracking_uri)
            set_experiment(experiment_name)
            
            # Ensure confusion matrix directory exists
            conf_dir = os.path.join('artifacts', 'confussion')
            os.makedirs(conf_dir, exist_ok=True)

            # Load dataset once (from project root artifacts)
            dataset_path = os.path.join('artifacts', 'data.csv')
            dataset_df = None
            if os.path.exists(dataset_path):
                dataset_df = pd.read_csv(dataset_path)
            
            # Log each model run
            for model_name, best_model in best_models.items():
                report = reports[model_name]
                report_dict = report["report_dict"]
                best_params = report.get("best_params", {})
                metrics = self._flatten_report(report_dict)
                
                # Predictions for confusion matrix
                y_pred = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"{model_name} Confusion Matrix")
                img_name = f"{model_name.lower().replace(' ', '_')}_confusion.jpg"
                img_path = os.path.join(conf_dir, img_name)
                plt.savefig(img_path, format='jpg', dpi=200, bbox_inches='tight')
                plt.close()
                
                # MLflow logging
                with start_run(run_name=model_name):
                    # Log dataset to this run
                    if dataset_df is not None:
                        log_pandas_dataset(dataset_df, name="Breast_Cancer", targets="diagnosis")
                    if best_params:
                        log_params(best_params)
                    if metrics:
                        log_metrics(metrics)
                    # Log model with flavor
                    if isinstance(best_model, LogisticRegression):
                        log_sklearn_model(best_model, name=f"{model_name}_model")
                    elif isinstance(best_model, xgb.XGBClassifier):
                        log_xgboost_model(best_model, name=f"{model_name}_model")
                    else:
                        log_sklearn_model(best_model, name=f"{model_name}_model")
                    # Log confusion matrix artifact
                    log_artifact(img_path)
            
            # Save the best-by-accuracy model and its report locally
            best_name = None
            best_acc = -1.0
            for model_name, rep in reports.items():
                try:
                    acc = float(rep["report_dict"].get("accuracy", -1.0))
                except Exception:
                    acc = -1.0
                if acc > best_acc:
                    best_acc = acc
                    best_name = model_name
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    "model": best_models.get(best_name),
                    "report": reports.get(best_name)
                }
            )
            logging.info("Best model saved")
            
            return reports
            
        except Exception as e:
            raise CustomException(e, sys)
            logging.info("Exception occurred during model training")
        
        logging.info("Model training completed")