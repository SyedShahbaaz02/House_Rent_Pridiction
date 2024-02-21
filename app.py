import pickle
import flask
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/carridge.pkl','rb'))
# standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        BHK=float(request.form.get('BHK'))
        Size = float(request.form.get('Size'))
        City = float(request.form.get('City'))
        Bathroom = float(request.form.get('Bathroom'))


        # new_data_scaled=standard_scaler.transform([[BHK,Size,City,Bathroom]])
        result=ridge_model.predict([[BHK,Size,City,Bathroom]])

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
