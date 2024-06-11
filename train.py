from src.components.data_loader import DataLoader
from src.components.data_transformer import DataTransformer
from src.components.model_trainer import ModelTrainer

data_path = "notebooks/WomenDHS.csv"

data = DataLoader(read_from_path=data_path)
data.load()
train_data, test_data = data.split(test_size=0.2, random_state=42)

target_variable = "WealthIndex"
numerical_inputs = ['Age', 'NumChild']
categorical_inputs = ['Residence','HouseHeadSex', 'Education']

data_transformer = DataTransformer(target_variable, numerical_inputs, categorical_inputs)
X_train, y_train, X_test, y_test = data_transformer.transform(train_data, test_data)
data_transformer.save_transformer()


from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    "ElasticNet":{"alpha": [0.1, 0.5, 1.0, 5.0, 10.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "K-Neighbors":{"n_neighbors": [3, 5, 7, 9, 11]},
    "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
    "Random Forest":{'n_estimators': [8, 16, 32, 64, 128, 256]},
    "XGBoost":{'learning_rate': [0.1, 0.01, 0.05, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
    "CatBoosting":{'depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100]},
    "AdaBoost":{'learning_rate': [0.1, 0.01, 0.05, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
    "Gradient Boosting":{'learning_rate':[0.1, 0.01, 0.05, 0.001],
                            'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                            'n_estimators': [8, 16, 32, 64, 128, 256]},
                            }

models = ModelTrainer(candidate_models, hyper_params=hyper_params)

models.train(X_train, y_train)
models.save_trained_models()

models.test(X_test, y_test)
models.save_train_test_scores()



