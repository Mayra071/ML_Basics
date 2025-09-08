import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
from src.Breast_Cancer_prediction.exception import CustomException
from src.Breast_Cancer_prediction.logger import logging
from src.Breast_Cancer_prediction.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try:
            logging.info("Data Transformation initiated")

            # Numerical columns only (target will be encoded separately)
            numerical_cols=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                            'smoothness_mean', 'compactness_mean', 'concavity_mean',
                            'concave points_mean', 'symmetry_mean',
                            'fractal_dimension_mean', 'radius_se', 
                            'texture_se', 'perimeter_se', 'area_se', 
                            'smoothness_se', 'compactness_se', 'concavity_se', 
                            'concave points_se', 'symmetry_se','fractal_dimension_se', 
                            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                            'smoothness_worst', 'compactness_worst', 'concavity_worst',
                            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst' 
                            ]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])

            # Combine both
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_cols)
            ])

            logging.info("Numerical and categorical columns are transformed")
            return preprocessor

        except Exception as e:
            logging.info("Exception occurred in data transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            # Preprocessor
            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            # Target column
            target_column_name = 'diagnosis'

            # Drop unwanted columns
            drop_columns = ['id', 'Unnamed: 32', target_column_name]

            # Split train data
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name].map({'B': 0, 'M': 1}).astype(int)

            # Split test data
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name].map({'B': 0, 'M': 1}).astype(int)

            logging.info("Applying preprocessing object on training and testing datasets")

            # Transform data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            # Final shape of arrays
            logging.info(f"Final Train Array Shape: {train_arr.shape}")
            logging.info(f"Final Test Array Shape: {test_arr.shape}")

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)
