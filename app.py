from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Pipe.prediction_pipeline import custom_data,predict_pipeline

application = Flask(__name__)

app = application

# Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data = custom_data(
            clonesize = request.form.get('clonesize'),
            honeybee = request.form.get('honeybee'),
            bumbles = request.form.get('bumbles'),
            andrena = request.form.get('andrena'),
            osmia = request.form.get('osmia'),
            MaxOfUpperTRange = request.form.get('MaxOfUpperTRange'),
            MinOfUpperTRange = request.form.get('MinOfUpperTRange'),
            AverageOfUpperTRange = request.form.get('AverageOfUpperTRange'), 
            MaxOfLowerTRange = request.form.get('MaxOfLowerTRange'), 
            MinOfLowerTRange = request.form.get('MinOfLowerTRange'),
            AverageOfLowerTRange = request.form.get('AverageOfLowerTRange'), 
            RainingDays = request.form.get('RainingDays'),
            AverageRainingDays = request.form.get('AverageRainingDays'),
            fruitset = request.form.get('fruitset'),
            fruitmass = request.form.get('fruitmass'), 
            seeds = request.form.get('seeds')
            )
        pred_df = data.get_data_as_dataframe()
        
        predict_pipelinex = predict_pipeline()
        result = predict_pipelinex.predict(pred_df)
        
        return render_template('home.html', results=result[0])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
