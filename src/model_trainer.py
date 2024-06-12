import os
import sys
import pandas as pd

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_model
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, candidate_models, hyper_params=None):
        self.candidate_models = candidate_models
        self.hyper_params = hyper_params
        self.are_models_trained = False
        self.are_models_tested = False

    def train(self, X_train, y_train):
        try:
            train_scores = []
            for (name, model) in self.candidate_models.items():

                if self.hyper_params is not None:
                    hyper_params = self.hyper_params[name]
                    grid_search = GridSearchCV(model, hyper_params, cv=3)
                    grid_search.fit(X_train, y_train)
                    model.set_params(**grid_search.best_params_)     

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)

                train_mae, train_mse, train_r2 = evaluate_model(y_train, y_train_pred)
                train_scores.append({"Model": name, "Train_MAE": train_mae, "Train_MSE": train_mse, "Train_R2": train_r2})
            
            logging.info("Model(s) trained.")

            self.are_models_trained = True
            self.train_scores_df = pd.DataFrame(train_scores)

            return self.train_scores_df

        except Exception as e:
            raise CustomException(e, sys)

    def save_trained_models(self, save_to_folder="trained_models"):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")
            
            models_saving_folder = f"models/{save_to_folder}"
            os.makedirs(models_saving_folder, exist_ok=True)

            for model_type, trained_model in zip(self.candidate_models.keys(), self.candidate_models.values()):
                trained_model_saving_path = os.path.join(models_saving_folder, f"{model_type}.pkl")
                save_object(file_path=trained_model_saving_path, object=trained_model)
            
            logging.info(f'Trained models saved in "{models_saving_folder}".')

        except Exception as e:
            raise CustomException(e, sys)


    def test(self, X_test, y_test):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")

            test_scores = []

            for (name, model) in self.candidate_models.items():
                y_test_pred = model.predict(X_test)
                test_mae, test_mse, test_r2 = evaluate_model(y_test, y_test_pred)

                test_scores.append({"Model": name, "Test_MAE": test_mae, "Test_MSE": test_mse, "Test_R2": test_r2})
                
            logging.info("Model(s) tested.")
            self.are_models_tested = True
            self.test_scores_df = pd.DataFrame(test_scores)

            return self.test_scores_df

        except Exception as e:
            raise CustomException(e, sys)

    def save_train_test_scores(self, save_to_folder="models"):
        try:
            if self.are_models_trained == False:
                raise ValueError("No model is trained.")
            if self.are_models_tested == False:
                raise ValueError("No model is tested.")

            os.makedirs(save_to_folder, exist_ok=True)

            scores_path = os.path.join(save_to_folder, "train_test_scores.csv")            

            scores_df = pd.merge(self.train_scores_df, self.test_scores_df, on="Model")
            # results_df = results_df.sort_values(by=["Test_R2"],ascending=False)
            scores_df.to_csv(scores_path, index=False, header=True)
            logging.info(f'Model(s) evaluations saved in "{save_to_folder}".')

            return scores_path

        except Exception as e:
            raise CustomException(e, sys)

