from flask import Flask, render_template,request
import pickle as pkl
import pandas as pd
import numpy as np




model = pkl.load(open('modelrfc.pkl', 'rb'))

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
        avg_glucose_level= float(request.form['glucose'])
        smoking_status= request.form['smoking-status']

        #gender
        if gender == 'male':
            gender = 1
        else:
            gender = 0  



        #married
        if married == 'yes':
            ever_married = 1
        else:
            ever_married = 0

        #work_type
        if work_type == 'private':
            work_type_Never_worked = 0
            work_type_Private = 1 
            work_type_Self_employed = 0
            work_type_children = 0  
            work_type_Govt_job = 0 
        elif work_type == 'self_employed':
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
            work_type_Govt_job = 0
        elif work_type == 'children':
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
            work_type_Govt_job = 0
        elif work_type == 'never_worked':
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0 
            work_type_Govt_job = 0
        elif work_type == 'govt_job':
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0    
            work_type_Govt_job = 1
        else:    
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 0

        #residence
        if resident_type == 'urban':
            Residence_type_Urban = 1
            Residence_type_Rural =0
        elif resident_type == 'rural':
            Residence_type_Urban = 0
            Residence_type_Rural =1
        else:
            Residence_type_Urban = 0
            Residence_type_Rural =0

        # smoking status
        if smoking_status == 'smokes':
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 1
            smoking_status_Unknown = 0
        elif (smoking_status == 'never_smoked'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_smokes = 0 
            smoking_status_Unknown = 0
        elif (smoking_status == 'formerly_smoked'):
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0 
            smoking_status_Unknown = 0
        elif (smoking_status == 'unknown'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0     
            smoking_status_Unknown = 1
        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0 
            smoking_status_Unknown = 0



        prediction = model.predict([[gender, age, hypertension, heart_disease, ever_married,avg_glucose_level, bmi,
            work_type_Govt_job,work_type_Never_worked, work_type_Private,work_type_Self_employed, work_type_children, 
            Residence_type_Rural,Residence_type_Urban, smoking_status_Unknown,smoking_status_formerly_smoked, 
            smoking_status_never_smoked,smoking_status_smokes]])[0]

        if prediction == 1:
            prediction = " Stroke"
        else:
            prediction = " No Stroke"    

        return render_template('output.html', data= prediction) 

    else:
        return render_template('index.html')
        



if __name__ == "__main__":
    app.run(debug=True)
