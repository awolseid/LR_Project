import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import data_preprocessor, save_object

class DataTransformation:
    def __init__(self, target_var, numerical_inputs, categorical_inputs):
        self.target_variable = list(target_var)
        self.numerical_inputs = numerical_inputs
        self.categorical_inputs = categorical_inputs
        self.data_transformer = None

    def transform_data(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read.")
            inputs_train_df = train_df.filter(self.numerical_inputs + self.categorical_inputs)
            inputs_test_df = test_df.filter(self.numerical_inputs + self.categorical_inputs)
            
            self.data_transformer = data_preprocessor(self.numerical_inputs, self.categorical_inputs)

            inputs_train_array = self.data_transformer.fit_transform(inputs_train_df)
            inputs_test_array = self.data_transformer.transform(inputs_test_df)
            logging.info(f"Inputs {self.numerical_inputs} scaled and Inputs {self.categorical_inputs} encoded.")

            target_train_array = np.array(train_df.filter(self.target_variable))
            target_test_array = np.array(test_df.filter(self.target_variable))

            train_array = np.c_[inputs_train_array, target_train_array] 
            test_array = np.c_[inputs_test_array, target_test_array] 

            return (train_array, test_array)
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_preprocessor(self, saving_folder):
        try:
            if self.data_transformer is None:
                raise ValueError("Data_transformer to be saved is None.")
            
            preprocessor_saving_path = os.path.join(saving_folder, "preprocessor.pkl")
            save_object(file_path=preprocessor_saving_path, object=self.data_transformer)
            logging.info(f"Pickle file of data preprocessor saved.")

            return preprocessor_saving_path

        except Exception as e:
            raise CustomException(e, sys)
