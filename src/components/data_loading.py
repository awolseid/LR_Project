
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class GetData:
    def __init__(self, data_to_be_saved_folder:str="input_data"):
        self.folder=data_to_be_saved_folder
        
        self.raw_data_path=os.path.join(self.folder, "data.csv")
        self.train_data_path=os.path.join(self.folder, "train.csv")
        self.test_data_path=os.path.join(self.folder, "test.csv")

    def import_data(self):
        try:
            df=pd.read_csv("notebook\data\ImpactCOVID.csv")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            df.to_csv(self.raw_data_path, index=False, header=True)   
            logging.info(f'Data imported and saved in "{self.folder}" folder.')        

            train_data, test_data=train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-Test split completed.")

            train_data.to_csv(self.train_data_path, index=False, header=True)
            test_data.to_csv(self.test_data_path, index=False, header=True)
            logging.info(f'Loaded data saved in "{self.folder}" folder.')
    
            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    data = GetData()
    data.import_data()



