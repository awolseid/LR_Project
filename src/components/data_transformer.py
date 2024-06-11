import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import data_transformation, save_object

class DataTransformer:
    def __init__(self, target_var: str, numerical_inputs: list, categorical_inputs: list):
        self.target_variable = target_var
        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs
        self.data_transformer = None

    def transform(self, train_df, test_df):
        try:
            X_train_df = train_df[self.numerical_inputs + self.categorical_inputs]
            y_train_array = np.array(train_df[self.target_variable])

            X_test_df = test_df[self.numerical_inputs + self.categorical_inputs]
            y_test_array = np.array(test_df[self.target_variable])
            
            self.data_transformer = data_transformation(self.numerical_inputs, self.categorical_inputs)

            X_train_array = self.data_transformer.fit_transform(X_train_df)
            X_test_array = self.data_transformer.transform(X_test_df)
            logging.info(f"Inputs {self.numerical_inputs} scaled and Inputs {self.categorical_inputs} encoded.")
     
            return (X_train_array, y_train_array, X_test_array, y_test_array)
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_transformer(self, save_to_folder="transformer"):
        try:
            if self.data_transformer is None:
                raise ValueError("No data transformer.")
            
            transformer_saving_folder = f"data/{save_to_folder}"
            os.makedirs(transformer_saving_folder, exist_ok=True)

            transformer_saving_path = os.path.join(transformer_saving_folder, "data_transformer.pkl")
            
            save_object(file_path=transformer_saving_path, object=self.data_transformer)
            logging.info(f'Data transformer saved in "{transformer_saving_folder}".')

            return transformer_saving_path

        except Exception as e:
            raise CustomException(e, sys)
