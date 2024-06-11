import pandas as pd
from prediction.prediction import Prediction

transformer_path = "data/transformer/data_transformer.pkl"
trained_model_path = "models/trained_models/Gradient Boosting.pkl"

predictor = Prediction(transformer_path, trained_model_path)

obs = pd.DataFrame({'Age': [42], 'NumChild': [5],
       'Residence': ["Urban"], 'HouseHeadSex': ["Male"], 'Education': ["Complete Secondary"]})

predictor.predict(inputs=obs)



from flask import Flask, request, render_template

app = Flask(__name__, template_folder='prediction/templates')

@app.route("/")
def index():
    return render_template("index.html") # searches for "templates/index.html"

@app.route("/make_prediction", methods=["GET", "POST"])
def predict_output():
    if request.method=="GET":
        return render_template("home.html")
    else:
        inputs_df = pd.DataFrame({"Residence": [request.form.get("Residence")],
                                  "Education": [request.form.get("Education")],
                                  "HouseHeadSex": [request.form.get("HouseHeadSex")],
                                  "Age": [float(request.form.get("Age"))],
                                  "NumChild": [float(request.form.get("NumChild"))]})
        print(inputs_df)

        predictor = Prediction(transformer_path, trained_model_path)
        predicted_output = predictor.predict(inputs_df)
        return render_template("home.html", result=predicted_output)


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)




