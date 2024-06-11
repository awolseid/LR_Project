import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class DataLoader:
    def __init__(self, read_from_path):
        self.read_from_path = read_from_path
        self.is_data_imported = False
        self.is_data_split = False

    def load(self, save_to_folder="raw"): 
        try:
            self.raw_data_df = pd.read_csv(self.read_from_path)
            logging.info(f'Data loaded.')
            self.is_data_imported = True

            raw_data_folder = f"data/{save_to_folder}"
            os.makedirs(raw_data_folder, exist_ok=True)

            raw_data_path = os.path.join(raw_data_folder, "raw_data.csv")
     
            self.raw_data_df.to_csv(raw_data_path, index=False, header=True)   
            
            logging.info(f'Raw data saved in "{raw_data_folder}".')  

            return self.raw_data_df
        
        except Exception as e:
            raise CustomException(e, sys)

    def clean(self):
        pass

    def split(self, test_size, random_state, save_to_folder="train_test"):
        try: 
            if self.is_data_imported == False:
                raise ValueError("No data imported.")

            self.train_data, self.test_data = train_test_split(self.raw_data_df, test_size=test_size, random_state=random_state)
            self.is_data_split = True
            logging.info("Train-test split done.")

            train_test_data_folder = f"data/{save_to_folder}"
            os.makedirs(train_test_data_folder, exist_ok=True)

            train_data_path = os.path.join(train_test_data_folder, "train_data.csv")
            test_data_path = os.path.join(train_test_data_folder, "test_data.csv")
            
            self.train_data.to_csv(train_data_path, index=False, header=True)   
            self.test_data.to_csv(test_data_path, index=False, header=True)   
            logging.info(f'Train and test data saved in "{train_test_data_folder}".')  
    
            return (self.train_data, self.test_data)

        except Exception as e:
            raise CustomException(e, sys)



