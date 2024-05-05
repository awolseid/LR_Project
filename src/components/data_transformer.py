import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import data_preprocessor, save_object

class Transformer:
    def __init__(self, target_var: str, numerical_inputs: list, categorical_inputs: list):
        self.target_variable = target_var
        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs
        self.data_transformer = None

    def transform_data(self, train_df, test_df):
        try:
            inputs_train_df = train_df[self.numerical_inputs + self.categorical_inputs]
            inputs_test_df = test_df[self.numerical_inputs + self.categorical_inputs]
            
            self.data_transformer = data_preprocessor(self.numerical_inputs, self.categorical_inputs)
            inputs_train_array = self.data_transformer.fit_transform(inputs_train_df)
            inputs_test_array = self.data_transformer.transform(inputs_test_df)
            logging.info(f"Inputs {self.numerical_inputs} scaled and Inputs {self.categorical_inputs} encoded.")

            target_train_array = np.array(train_df[self.target_variable])
            target_test_array = np.array(test_df[self.target_variable])

            return (inputs_train_array, target_train_array, inputs_test_array, target_test_array)
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_transformer(self, saving_folder):
        try:
            if self.data_transformer is None:
                raise ValueError("No data transformer.")
            
            transformer_saving_path = os.path.join(saving_folder, "preprocessor.pkl")
            save_object(file_path=transformer_saving_path, object=self.data_transformer)
            logging.info(f"Pickle file of data preprocessor saved in {saving_folder} folder.")

            return transformer_saving_path

        except Exception as e:
            raise CustomException(e, sys)
