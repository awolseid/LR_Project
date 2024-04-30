
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

class GetData:
    def __init__(self):
        self.raw_data_path: str=os.path.join("inputdata","data.csv")
        self.train_data_path: str=os.path.join("inputdata","train.csv")
        self.test_data_path: str=os.path.join("inputdata","test.csv")

    def import_data(self):
        try:
            df=pd.read_csv("notebook\data\ImpactCOVID.csv")
            logging.info("Data imported.")

            df.to_csv(self.raw_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            train_data, test_data=train_test_split(df, test_size=0.2, random_state=42)
            loggin.info("Train-Test split completed.")

            train_data.to_csv(self.train_data_path, index=False, header=True)
            test_data.to_csv(self.test_data_path, index=False, header=True)
            loggin.info("Train-Test data saved.")
            loggin.info("Data importing completed.")
            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    data = GetData()
    data.import_data()



