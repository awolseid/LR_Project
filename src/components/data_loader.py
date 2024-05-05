
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class Data:
    def __init__(self):
        self.is_data_imported = False
        self.is_data_split = False

    def import_data(self, reading_data_path): 
        try:
            data_df = pd.read_csv(reading_data_path)
            logging.info(f'Data imported.')
            self.is_data_imported = True
            return data_df
        
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, data_df, test_size, random_state):
        try: 
            if self.is_data_imported == False:
                raise ValueError("No data imported.")

            self.train_data, self.test_data = train_test_split(data_df, test_size=test_size, random_state=random_state)
            self.is_data_split = True
            logging.info("Data split to Train and Test.")
    
            return (self.train_data, self.test_data)

        except Exception as e:
            raise CustomException(e, sys)

    def save_split_data(self, saving_folder):
        try: 
            if self.is_data_imported == False:
                raise ValueError("No data is imported.")

            if self.is_data_split == False:
                raise ValueError("No train and test split data.")

            artifacts = f"artifacts/{saving_folder}"
            os.makedirs(artifacts, exist_ok=True)

            train_data_path = os.path.join(artifacts, "train.csv")
            test_data_path = os.path.join(artifacts, "test.csv")
            
            self.train_data.to_csv(train_data_path, index=False, header=True)   
            self.test_data.to_csv(test_data_path, index=False, header=True)   
            logging.info(f'Training and testing data saved in "{artifacts}".')  
    
            return (train_data_path, test_data_path)

        except Exception as e:
            raise CustomException(e, sys)



