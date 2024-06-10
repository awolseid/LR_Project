import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self, preprocessor_path, model_path):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, residence, education, househeadsex, age, numchild):
        self.residence = residence
        self.education = education
        self.househeadsex = househeadsex
        self.age = age
        self.numchild = numchild
    
    def get_data_frame(self):
        try:
            input_data_dict = {
                "Residence": [self.residence],
                "Education": [self.education],
                "HouseHeadSex": [self.househeadsex],
                "Age": [self.age],
                "NumChild": [self.numchild]
                }
            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
        



