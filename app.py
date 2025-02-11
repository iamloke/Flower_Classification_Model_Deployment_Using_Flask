import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('/home/loke/repo/ai_project/Flower_Classification/ML_MODEL/model.pkl')

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text = f"The flower species is {prediction}")

if __name__ == "__main__":
    app.run(debug=True)