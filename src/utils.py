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



def split_data(df:pd.DataFrame, test_size, random_state):
    try:
        train_data, test_data = train_test_split(df, test_size, random_state)
        return (train_data, test_data)

    except Exception as e:
        raise CustomException(e, sys)


def data_preprocessor(numerical_inputs, categorical_inputs):
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
