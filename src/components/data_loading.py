
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation

class GetData:
    def __init__(self, saving_folder):
        self.saving_folder = saving_folder     
        self.raw_data_path = os.path.join(self.saving_folder, "data.csv")
        self.train_data_path = os.path.join(self.saving_folder, "train.csv")
        self.test_data_path = os.path.join(self.saving_folder, "test.csv")

    def import_data(self, reading_data_path):
        try:
            df = pd.read_csv(reading_data_path)

            os.makedirs(self.saving_folder, exist_ok=True) 
            #### os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)   
            logging.info(f'Data imported and saved in "{self.saving_folder}" folder.')        

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split to Train and Test.")

            train_data.to_csv(self.train_data_path, index=False, header=True)
            test_data.to_csv(self.test_data_path, index=False, header=True)
            logging.info(f'Train & Test data saved in "{self.saving_folder}" folder.')
    
            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    reading_data_path = "notebook\data\ImpactCOVID.csv"
    saving_folder = "input_data"
    
    obj = GetData(saving_folder=saving_folder)
    train_path, test_path = obj.import_data(reading_data_path=reading_data_path)

    data_transformation = DataTransformation()
    data_transformation.transform_data(train_path, test_path)





