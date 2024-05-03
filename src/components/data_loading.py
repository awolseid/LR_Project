
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation

class DataImporter:
    def __init__(self, reading_data_path):
        self.reading_data_path = reading_data_path
        self.imported_data = None

    def import_data(self): 
        try:
            self.imported_data = pd.read_csv(reading_data_path)
            logging.info(f'Data imported.') 
            return self.imported_data
        
        except Exception as e:
            raise CustomException(e, sys)

    def save_data(self, saving_folder):
        try:
            raw_data_path = os.path.join(saving_folder, "data.csv")
            train_data_path = os.path.join(saving_folder, "train.csv")
            test_data_path = os.path.join(saving_folder, "test.csv")
            
            self.imported_data.to_csv(raw_data_path, index=False, header=True)   
            logging.info(f'Imported data saved in "{saving_folder}" folder.')        

            train_data, test_data = train_test_split(self.imported_data, test_size=0.2, random_state=42)
            logging.info("Data split to Train and Test.")

            train_data.to_csv(train_data_path, index=False, header=True)
            test_data.to_csv(test_data_path, index=False, header=True)
            logging.info(f'Train & Test data saved in "{saving_folder}" folder.')
    
            return (train_data_path, test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    reading_data_path = "notebook\data\ImpactCOVID.csv"
    data_saving_folder = "input_data"
    
    obj = DataImporter(reading_data_path=reading_data_path)
    data_df = obj.import_data()

    train_path, test_path = obj.save_data(saving_folder=data_saving_folder)

    target_variable = "EconomicImpact"
    numerical_inputs = ['Age', 'NumRooms']
    categorical_inputs = ['Sex', 'Marital', 'Education', 'Employment', 'Income']

    data_transformation = DataTransformation(target_variable, numerical_inputs, categorical_inputs)
    transformed_train_array, transformed_test_array = data_transformation.transform_data(train_path, test_path)

    processor_saving_folder = "input_processor"
    preprocessor_saved_path = data_transformation.save_preprocessor(saving_folder=processor_saving_folder)


