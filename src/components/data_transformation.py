import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformation:
    def __init__(self, preprocessor_folder="input_preprocessor"):
        self.folder=preprocessor_folder
        self.preprocessor_file_path=os.path.join(self.folder, "preprocessor.pkl")
    
    def get_transformer(self):

        try:
            numerical_columns=['Age', 'NumRooms']
            categorical_columns=['Sex', 'Marital', 'Education', 'Employment', 'Income']

            num_processing=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns {numerical_columns} scaled.")

            cat_processing=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns {categorical_columns} encoded.")

            preprocessor=ColumnTransformer(
                [
                    ("num_processing", num_processing, numerical_columns),
                    ("cat_processing", cat_processing, categorical_columns)
                    ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def transform_data(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and test data read.")


            numerical_columns=['Age', 'NumRooms']
            categorical_columns=['Sex', 'Marital', 'Education', 'Employment', 'Income']
            
            inputs_train_df=train_df.filter(numerical_columns + categorical_columns)
            target_train_df=train_df.filter(["EconomicImpact"])

            inputs_test_df=test_df.filter(numerical_columns + categorical_columns)
            target_test_df=test_df.filter(["EconomicImpact"])

            data_transformer=self.get_transformer()
            inputs_tain_array=data_transformer.fit_transform(inputs_train_df)
            inputs_test_array=data_transformer.transform(inputs_test_df)

            """
            fit_transform() on training set calculates scaling parameters and apply transformation.
            transform() on test set apply same transformation using scaling parameters from training set.
            """

            train_array=np.c_[inputs_tain_array, np.array(target_train_df)] # concatenating columnwise
            test_array=np.c_[inputs_test_array, np.array(target_test_df)] 

            logging.info("Processed data saved.")

            save_object(file_path=self.preprocessor_file_path, obj=data_transformer)

            logging.info("Pickle file of processor saved.")

            return (train_array, test_array, self.preprocessor_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)









