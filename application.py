import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

# FIXED: Added 'GET' to methods so the page can load in the browser
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # FIXED: Keys here now perfectly match the "name" attributes in home.html
        data = CustomData(
            gender=request.form['gender'],
            race_ethnicity=request.form['race_ethnicity'],
            parental_level_of_education=request.form['parental_level_of_education'],
            lunch=request.form['lunch'],
            test_preparation_course=request.form['test_preparation_course'],
            reading_score=int(request.form['reading_score']),
            writing_score=int(request.form['writing_score'])
        )
        print("======BUTTON WAS CLICKED! HERE IS THE DATA:======")
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        # Passes 'result' to match your HTML template variable {{result}}
        return render_template('home.html', result=result[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)