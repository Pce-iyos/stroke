

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

import pickle


df = pd.read_csv('dissertation-stroke-data.csv')

df2= df.drop(['id'],axis = 1)


age_grouping=[]
for a in df['age']:
    if a<14.0:
        age_grouping.append('Children')
    elif a>14.0 and a<=24.0:
        age_grouping.append('Youths')
    elif a>24.0 and a<64.0:
        age_grouping.append('Adults')
    else:
        age_grouping.append('Elderly')
df['age_grouping']=age_grouping

def impute_missing_bmi(cols):
    bmi = cols[0]
    age_grouping = cols[1]
    gender = cols[2]
    if pd.isnull(bmi):
        if age_grouping == 'Elderly':
            if gender == 'Male':
                return 28.7
            else:
                return 28.8
        elif age_grouping == 'Adults':
            if gender == 'Male':
                return 30.6
            else:
                return 29.1
        elif age_grouping == 'Youths':
            if gender == 'Male':
                return 25.1
            else:
                return 25.7
        else:
            if gender == 'Male':
                return 18.7
            else:
                return 18.6
    else:
        return bmi
df['bmi']=df[['bmi','age_grouping','gender']].apply(impute_missing_bmi,axis=1)



bmi_grouping=[]
for b in df['bmi']:
    if b<18.0:
        bmi_grouping.append('Underweight')
    elif b>18.0 and a<=25.0:
        bmi_grouping.append('Normal_weight')
    elif a>25.0 and a<30.0:
        bmi_grouping.append('Overweight')
    else:
        bmi_grouping.append('Obese')
df['bmi_grouping']=bmi_grouping






df3 = df.loc[(df['avg_glucose_level']>250)]



glucose_grouping=[]
for c in df['avg_glucose_level']:
    if c<150.0:
        glucose_grouping.append('Normal')
    elif c>150.0 and c<=200.0:
        glucose_grouping.append('Pre-diabetic')
    else:
        glucose_grouping.append('Diabetic')
df['glucose_grouping']=glucose_grouping
df.head()



df.drop(index = df[df['avg_glucose_level']>250].index[0] ,axis=0,inplace=True)


# encoding
# create a list of call categorical variables

df_encode=pd.get_dummies(df[['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']], drop_first=True)
df_encode.head()


# to merge the encoded variables with the other variables
df[df_encode.columns]=df_encode
df.head()



df.drop(['gender','ever_married','work_type','Residence_type','smoking_status'],axis=1,inplace=True)



df_model=df.drop(['id','age_grouping','bmi_grouping','glucose_grouping'],axis=1)
df_model.head()



#for the oversampling
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X=df_model.drop(['stroke'],axis=1)
y=df_model['stroke']
X_over, y_over = oversample.fit_resample(X, y)




# to split data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,test_size=0.3,stratify=y_over,random_state=42)



# scaling the numerical variables
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num = ['age','bmi','avg_glucose_level']
X_train[num]= scaler.fit_transform (X_train[num])
X_test[num]= scaler.transform (X_test[num])
X_train.head()



#import the relevant library for the model development and evaluation metric
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 




#logistic regression
import warnings 
warnings.filterwarnings('ignore')
param_grid={'C':[0.001,0.01,0.1,1,10,100], 'max_iter':[50,75,100,200,300,400,500,700]}
logreg=RandomizedSearchCV(LogisticRegression(solver='lbfgs'),param_grid,cv=5)
logreg.fit(X_train,y_train)
y_pred_logreg=logreg.predict(X_test)
confusion_logreg=confusion_matrix(y_test,logreg.predict(X_test))
plt.figure(figsize=(7,7))
sns.heatmap(confusion_logreg,annot=True)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
print(classification_report(y_test,y_pred_logreg))




# Multi layered perceptron neural network

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


#Create an MLP model with 4 hidden layers with varying number of perceptrons in each layer, number of iterations is 500
mlp = MLPClassifier(hidden_layer_sizes=(2,8,9,9),max_iter = 2000)
classifier = mlp.fit(X_train,y_train)
y3_pred_mlp = mlp.predict(X_test)
confusion_mlp = confusion_matrix(y_test,mlp.predict(X_test))
plt.figure(figsize=(9,9))
sns.heatmap(confusion_mlp,annot=True)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
print(classification_report(y_test,y3_pred_mlp))




# Decision Tree Model

dtree =  DecisionTreeClassifier(random_state = 0, max_depth = 5)
d_tree = dtree.fit(X_train,y_train)
y3_pred_dtree = d_tree.predict(X_test)
confusion_dtree = confusion_matrix(y_test,dtree.predict(X_test))
plt.figure(figsize=(8,7))
sns.heatmap(confusion_dtree,annot=True)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
print(classification_report(y_test,y3_pred_dtree))



#SVM Model

svm=RandomizedSearchCV(SVC(),param_grid,cv=5)
svm.fit(X_train,y_train)
y_pred_svm=svm.predict(X_test)
confusion_svm=confusion_matrix(y_test,svm.predict(X_test))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_svm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_svm)) 



pickle.dump(d_tree,open('modeldtree.pkl', 'wb'))




