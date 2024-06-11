import os
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import dill

from src.exception import CustomException

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def data_transformation(numerical_inputs: list, categorical_inputs: list):
    try:
        numerical_preprocessing = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ])

        categorical_preprocessing = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder())
            ])

        transformer = ColumnTransformer([
            ("num_processing", numerical_preprocessing, numerical_inputs),
            ("cat_processing", categorical_preprocessing, categorical_inputs)
            ])
        return transformer

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, object):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(actual, predicted):
    try:
        MAE = mean_absolute_error(actual, predicted)
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)    
        return (MAE, MSE, R2)

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
