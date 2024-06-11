import sys
from src.exception import CustomException
from src.utils import load_object

class Prediction:
    def __init__(self, transformer_path, trained_model_path):
        self.transformer_path = transformer_path
        self.trained_model_path = trained_model_path

    def predict(self, inputs: dict):
        try:
            transformer = load_object(file_path=self.transformer_path)
            transformed_inputs = transformer.transform(inputs)

            trained_model = load_object(file_path=self.trained_model_path)
            predicted_output = trained_model.predict(transformed_inputs)[0]
            print(f"The predicted output is: {predicted_output}.")
            return predicted_output
        
        except Exception as e:
            raise CustomException(e, sys)
        