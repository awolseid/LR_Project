from src.components.data_loader import Data
from src.components.data_transformer import Transformer
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    reading_data_path = "notebook\data\ImpactCOVID.csv"
    
    data_object = Data()
    data_df = data_object.import_data(reading_data_path=reading_data_path)
    train_data, test_data = data_object.split_data(data_df, test_size=0.2, random_state=42)

    data_saving_folder = "input_data"
    train_data_path, test_data_path = data_object.save_split_data(saving_folder=data_saving_folder)

    target_variable = "EconomicImpact"
    numerical_inputs = ['Age', 'NumRooms']
    categorical_inputs = ['Sex', 'Marital', 'Education', 'Employment', 'Income']

    transformer = Transformer(target_variable, numerical_inputs, categorical_inputs)

    X_train, y_train, X_test, y_test = transformer.transform_data(train_data, test_data)

    transformer_saving_folder = "input_processor"
    transformer_saved_path = transformer.save_transformer(saving_folder=transformer_saving_folder)
    
    
    model_object = ModelTrainer()

    train_results_df = model_object.train_models(X_train, y_train)
    test_results_df = model_object.test_models(X_test, y_test)

    results_saving_folder = "models_summary"
    results_saved_path = model_object.save_train_test_results(saving_folder=results_saving_folder)

    trained_models_saving_folder = "trained_models"
    trained_models_saving_folder = model_object.save_trained_models(saving_folder=trained_models_saving_folder)