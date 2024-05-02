import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException



def split_data(df:pd.DataFrame, test_size, random_state):
    try:
        train_data, test_data = train_test_split(df, test_size, random_state)
        return (train_data, test_data)

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
