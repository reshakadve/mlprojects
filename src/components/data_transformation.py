import sys, os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        #This func is responsible for Data Transformation 
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")), #for missing values
                    ("scalar", StandardScaler(with_mean=False)) #scale the data 
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Cat Cols: {categorical_columns}")
            logging.info(f"Num Cols: {numerical_columns}")

            #combine num and cat pipeline using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
                
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            # Obtain the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()

            target_col = "math score"
            numerical_columns = ["writing score", "reading score"]

            # Split data into input features and target labels
            input_feature_train_df = train_df.drop(target_col, axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(target_col, axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessing object on training and testing data")

            # Apply the transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df)

            '''# Debugging: Check if the transformation contains any non-numeric data
            logging.info(f"Transformed Training Data Sample:\n{input_feature_train_arr[:5]}")
            logging.info(f"Transformed Testing Data Sample:\n{input_feature_test_arr[:5]}")'''

            # Combine the transformed input features with target labels
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object( 
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )


            return  (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            )

        except Exception as e:
            raise CustomException(e, sys)

     