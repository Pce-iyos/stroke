from pyexpat import features
from flask import Flask, render_template,request
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
model = pkl.load(open('modeldtree.pkl', 'rb'))

#Initialize the flask App
app = Flask(__name__)


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    #For rendering results on HTML GUI
    if (request.method == "POST"):
        gender= request.form['gender']
        age= int(request.form['age'])
        hypertension= int(request.form['hypertension'])
        heart_disease= request.form['heart-disease']
        married= request.form['ever-married']
        work_type= request.form['work-type']
        resident_type= request.form['resident-type']
        bmi= float(request.form['bmi'])
        avg_glucose= float(request.form['glucose'])
        smoking_status= request.form['smoking-status']

        #gender
        if gender == 'male':
            gender_Male = 1
        else:
            gender_Male = 0  



        #married
        if married == 'yes':
            ever_married_Yes = 1
        else:
            ever_married_Yes = 0

        #work_type
        if work_type == 'private':
            work_type_Never_worked = 0
            work_type_Private = 1 
            work_type_Self_employed = 0
            work_type_children = 0   
        elif work_type == 'self_employed':
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
        elif work_type == 'children':
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
        elif work_type == 'never_worked':
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0    
        else:    
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0

        #residence
        if resident_type == 'urban':
            Residence_type_Urban = 1
        else:
            Residence_type_Urban = 0

        # smoking status
        if smoking_status == 'smokes':
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 1
        elif (smoking_status == 'never_smoked'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_smokes = 0 
        elif (smoking_status == 'formerly_smoked'):
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0 
        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0 

        features = scaler.fit_transform([[age,hypertension,heart_disease,avg_glucose,bmi,gender_Male,ever_married_Yes,
        work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children, 
        Residence_type_Urban,smoking_status_formerly_smoked,smoking_status_never_smoked,smoking_status_smokes]])

        prediction = model.predict(features)[0]
        if prediction == 0:
            prediction = "Stroke Absent"
        else:
            prediction = "Stroke Present"    

        return render_template('output.html', data= prediction) 

    else:
        return render_template('index.html')
        



if __name__ == "__main__":
    app.run(debug=True)
