from src.components.data_loader import Data
from src.components.data_transformer import Transformer
from src.components.model_trainer import ModelTrainer

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

if __name__=="__main__":
    reading_data_path = "notebook\data\WomenDHS.csv"
    
    data_object = Data()
    data_df = data_object.import_data(reading_data_path=reading_data_path)
    train_data, test_data = data_object.split_data(data_df, test_size=0.2, random_state=42)

    data_saving_folder = "input_data"
    train_data_path, test_data_path = data_object.save_split_data(saving_folder=data_saving_folder)

    target_variable = "WealthIndex"
    numerical_inputs = ['Age', 'NumChild']
    categorical_inputs = ['Residence','HouseHeadSex', 'Education']

    transformer = Transformer(target_variable, numerical_inputs, categorical_inputs)

    X_train, y_train, X_test, y_test = transformer.transform_data(train_data, test_data)

    transformer_saving_folder = "input_processor"
    transformer_saved_path = transformer.save_transformer(saving_folder=transformer_saving_folder)
    
    candidate_models = {
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

    hyper_params = {
        "Linear":{},
        "Lasso":{"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]},
        "Ridge":{"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]},
        "ElasticNet":{
            "alpha": [0.1, 0.5, 1.0, 5.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
            },
        "K-Neighbors":{"n_neighbors": [3, 5, 7, 9, 11]},
        "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
        "Random Forest":{'n_estimators': [8, 16, 32, 64, 128, 256]},
        "XGBoost":{
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            },
        "CatBoosting":{
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
            },
        "AdaBoost":{
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            },
        "Gradient Boosting":{
            'learning_rate':[0.1, 0.01, 0.05, 0.001],
            'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            }
    
    model_object = ModelTrainer(candidate_models, hyper_params=hyper_params)

    train_results_df = model_object.train_models(X_train, y_train)
    test_results_df = model_object.test_models(X_test, y_test)

    results_saving_folder = "models_summary"
    results_saved_path = model_object.save_train_test_results(saving_folder=results_saving_folder)

    trained_models_saving_folder = "trained_models"
    trained_models_saving_folder = model_object.save_trained_models(saving_folder=trained_models_saving_folder)