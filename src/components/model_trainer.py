import os
import sys
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

class ModelTrainer:
    def __init__(self):
        self.candidate_models = {
            "Linear" : LinearRegression(),
            "Lasso" : Lasso(),
            "Ridge" : Ridge(),
            "ElasticNet" : ElasticNet(),
            "K-Neighbors" : KNeighborsRegressor(),
            "Decision Tree" : DecisionTreeRegressor(),
            "Random Forest" : RandomForestRegressor(),
            "XGBoost" : XGBRegressor(),
            "CatBoosting" : CatBoostRegressor(verbose=False),
            "AdaBoost": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
            }
        self.are_models_trained = False
        self.are_models_tested = False

        self.model_trainer = os.path.join("Model", "model.pkl")

    def train_models(self, X_train, y_train):
        try:
            train_results = []

            for (name, model) in self.candidate_models.items():
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)

                train_mae, train_mse, train_r2 = evaluate_model(y_train, y_train_pred)
                train_results.append({
                    "Model": name,
                    "Train_MAE": train_mae, 
                    "Train_MSE": train_mse, 
                    "Train_R2": train_r2})
            
            logging.info("Model(s) trained.")
            self.are_models_trained = True

            self.train_results_df = pd.DataFrame(train_results)

            return self.train_results_df

        except Exception as e:
            raise CustomException(e, sys)

    def test_models(self, X_test, y_test):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")

            test_results = []

            for (name, model) in self.candidate_models.items():
                y_test_pred = model.predict(X_test)
                test_mae, test_mse, test_r2 = evaluate_model(y_test, y_test_pred)

                test_results.append({
                    "Model": name,
                    "Test_MAE": test_mae, 
                    "Test_MSE": test_mse, 
                    "Test_R2": test_r2})
                
            logging.info("Model(s) tested.")
            self.are_models_tested = True
            self.test_results_df = pd.DataFrame(test_results)

            return self.test_results_df

        except Exception as e:
            raise CustomException(e, sys)

    def save_train_test_results(self, saving_folder: str):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")
            if self.are_models_tested == False:
                raise ValueError("No model is tested.")

            artifacts = f"artifacts/{saving_folder}"

            os.makedirs(artifacts, exist_ok=True)

            models_summary_path = os.path.join(artifacts, "train_test_results.csv")            

            results_df = pd.merge(self.train_results_df, self.test_results_df, on="Model").sort_values(by=["Test_R2"],ascending=False)
            results_df.to_csv(models_summary_path, index=False, header=True)
            logging.info(f'Model(s) evaluations saved in "{artifacts}".')

            return models_summary_path

        except Exception as e:
            raise CustomException(e, sys)

    def save_trained_models(self, saving_folder: str):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")
            
            artifacts = f"artifacts/{saving_folder}"
            os.makedirs(artifacts, exist_ok=True)

            for model_type, trained_model in zip(self.model_types, self.models):
                trained_model_saving_path = os.path.join(artifacts, f"{model_type}.pkl")
                save_object(file_path=trained_model_saving_path, object=trained_model)
            
            logging.info(f'Pickle files of trained models saved in "{artifacts}".')

            return f'Pickle files of trained models saved in "{artifacts}".'

        except Exception as e:
            raise CustomException(e, sys)
