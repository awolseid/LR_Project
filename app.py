from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

## Route for a home page

@app.route("/")
def index():
    return render_template("index.html") # search for "templates/index.html"

preprocessor_path = "artifacts/input_processor/preprocessor.pkl"
model_path =  "artifacts/trained_models/Gradient Boosting.pkl"

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data = CustomData(
            residence = request.form.get("Residence"),
            education = request.form.get("Education"),
            househeadsex = request.form.get("HouseHeadSex"),
            age = float(request.form.get("Age")),
            numchild = float(request.form.get("NumChild")),
        )
        features_df = data.get_data_frame()
        print(features_df)

        predict_pipeline = PredictPipeline(preprocessor_path, model_path)
        results = predict_pipeline.predict(features_df)
        return render_template("home.html", results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)




