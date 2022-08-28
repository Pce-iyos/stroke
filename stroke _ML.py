

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.metrics import plot_confusion_matrix,classification_report,roc_curve,plot_roc_curve,auc
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



data=pd.read_csv('dissertation-stroke-data.csv')



cat_columns = ['hypertension', 'heart_disease', 'stroke']
data[cat_columns] = data[cat_columns].astype(str)



#searching for duplicates in the dataset
print(f'Duplicates in data set: {data.duplicated().sum()}')




# Drop ID 
data.drop('id', axis= 1, inplace= True)


# categorical and numeric features
cat_features = data.select_dtypes(include="O").columns
num_cols = data.select_dtypes(include="number").columns
print('Categorical Features are: ', cat_features)
print('Numerical Features are: ', num_cols)



# ## Outlier Detection

fig, ax = plt.subplots(1, 3, figsize = (20, 5))
sns.boxplot(x = data['age'], ax= ax[0], color= 'blue', linewidth= 2)
sns.boxplot(x = data['avg_glucose_level'], ax= ax[1], color= 'blue', linewidth= 2)
sns.boxplot(x = data['bmi'], ax= ax[2], color= 'blue', linewidth= 2)


# ### Average glucose level and BMI has outliers

#handling the outliers
for col in ['avg_glucose_level', 'bmi']:
    data[col] = np.log(data[col])

fig,ax=plt.subplots(3,3,figsize=(25,15))

# Checking value in each categorical feature
cat_cols = cat_features[:-1]
for col in cat_cols:
    print(f'{col}\n {data[col].value_counts()}\n')



data.drop(data.loc[data['gender'] == 'Other'].index, inplace= True)





# Columns with missing values
missing_col=data.columns[data.isna().any()].tolist()

# Missing values
missing_num=pd.DataFrame(data[missing_col].isna().sum(), columns=['Number_missing'])
missing_num['Percentage_missing (%)']=np.round(100*missing_num['Number_missing']/len(data),2)
missing_num


data['bmi']=data['bmi'].fillna(data['bmi'].median())



##label encoding
categorical_cols = ['gender','hypertension','heart_disease', 'ever_married'] 

from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))   


# convert target variabe stroke into integer format 
data['stroke'] = data['stroke'].map({'0': 0, '1': 1})



## Onehot Encoding
data = pd.get_dummies(data, columns=['work_type', 'Residence_type', 'smoking_status'])
data.head()




X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'avg_glucose_level', 'bmi', 'work_type_Govt_job',
    'work_type_Never_worked', 'work_type_Private',
    'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
    'Residence_type_Urban', 'smoking_status_Unknown',
    'smoking_status_formerly smoked', 'smoking_status_never smoked',
    'smoking_status_smokes']]

y = data['stroke']

from imblearn.over_sampling import SMOTE

print ('Number of observations in the target variable before oversampling of the minority class:', np.bincount (y) )


fig, (ax1, ax2) = plt.subplots(1, 2)
sns.barplot(x=['0', '1'], y =[sum(y == 0), sum(y == 1)], ax = ax1)
ax1.set_title("Before Oversampling")
ax1.set_xlabel('Stroke')


smt = SMOTE ()
X_sam, y_sam = smt.fit_resample (X, y)

print ('\nNumber of observations in the target variable after oversampling of the minority class:', np.bincount (y_sam) )


sns.barplot(x=['0', '1'], y =[sum(y_sam == 0), sum(y_sam == 1)], ax = ax2)
ax2.set_title("After Oversampling")
ax2.set_xlabel('Stroke')

plt.tight_layout()
plt.show()




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X_sam, y_sam, test_size=0.2, random_state=20)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print(X_test.shape)
print(X_train.shape)


# ## Selecting Model

# ### RandomForest


rfc_clf = RandomForestClassifier(n_estimators=100,random_state=101)

rfc_clf.fit(X_train, y_train)


plot_confusion_matrix(rfc_clf, 
                    X_test, 
                    y_test, 
                    values_format='d', 
                    display_labels=['No Stroke', 'Stroke'])


train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = rfc_clf.score(X_train, y_train)
test_score = rfc_clf.score(X_test, y_test)
y_pred_rfc = rfc_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_rfc)
rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, y_pred_rfc)
test_auc = auc(rfc_fpr, rfc_tpr)
test_f1_score = f1_score(y_test,y_pred_rfc)
test_precision_score = precision_score(y_test,y_pred_rfc)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ## Xgboost


xgb_clf =XGBClassifier(learning_rate=0.1,objective='binary:logistic',random_state=0,eval_metric='mlogloss')

xgb_clf.fit(X_train, y_train)


plot_confusion_matrix(xgb_clf, 
                    X_test, 
                    y_test, 
                    values_format='d', 
                    display_labels=['No Stroke', 'Stroke'])


train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = xgb_clf.score(X_train, y_train)
test_score = xgb_clf.score(X_test, y_test)
y_pred_xgb = xgb_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_xgb)
xgb_fpr, xgb_tpr, thresholds = roc_curve(y_test, y_pred_xgb)
test_auc = auc(xgb_fpr, xgb_tpr)
test_f1_score = f1_score(y_test,y_pred_xgb)
test_precision_score = precision_score(y_test,y_pred_xgb)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ### Logistic Regression


log_clf =LogisticRegression(random_state=0)

log_clf.fit(X_train, y_train)


plot_confusion_matrix(log_clf, 
                    X_test, 
                    y_test, 
                    values_format='d', 
                    display_labels=['No Stroke', 'Stroke'])


train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = log_clf.score(X_train, y_train)
test_score = log_clf.score(X_test, y_test)
y_pred_log = log_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_log)
log_fpr, log_tpr, thresholds = roc_curve(y_test, y_pred_log)
test_auc = auc(log_fpr, log_tpr)
test_f1_score = f1_score(y_test,y_pred_log)
test_precision_score = precision_score(y_test,y_pred_log)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ### SVM

svm_clf =SVC(random_state=0)

svm_clf.fit(X_train, y_train)


plot_confusion_matrix(svm_clf, 
                    X_test, 
                    y_test, 
                    values_format='d', 
                    display_labels=['No Stroke', 'Stroke'])


train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = svm_clf.score(X_train, y_train)
test_score = svm_clf.score(X_test, y_test)
y_pred_svm = svm_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_svm)
svm_fpr, svm_tpr, thresholds = roc_curve(y_test, y_pred_svm)
test_auc = auc(svm_fpr, svm_tpr)
test_f1_score = f1_score(y_test,y_pred_svm)
test_precision_score = precision_score(y_test,y_pred_svm)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ### Catboost

cat_clf =CatBoostClassifier()

cat_clf.fit(X_train, y_train)


plot_confusion_matrix(cat_clf, 
                    X_test, 
                    y_test, 
                    values_format='d', 
                    display_labels=['No Stroke', 'Stroke'])



train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = cat_clf.score(X_train, y_train)
test_score = cat_clf.score(X_test, y_test)
y_pred_cat = cat_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_cat)
cat_fpr, cat_tpr, thresholds = roc_curve(y_test, y_pred_cat)
test_auc = auc(cat_fpr, cat_tpr)
test_f1_score = f1_score(y_test,y_pred_cat)
test_precision_score = precision_score(y_test,y_pred_cat)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ### Ada Boost


ada_clf =AdaBoostClassifier()

ada_clf.fit(X_train, y_train)


plot_confusion_matrix(ada_clf, 
        X_test, 
        y_test, 
        values_format='d', 
        display_labels=['No Stroke', 'Stroke'])



train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = ada_clf.score(X_train, y_train)
test_score = ada_clf.score(X_test, y_test)
y_pred_ada = ada_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_ada)
ada_fpr, ada_tpr, thresholds = roc_curve(y_test, y_pred_ada)
test_auc = auc(ada_fpr, ada_tpr)
test_f1_score = f1_score(y_test,y_pred_ada)
test_precision_score = precision_score(y_test,y_pred_ada)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)


# ### KNN

knn_clf =KNeighborsClassifier()

knn_clf.fit(X_train, y_train)


plot_confusion_matrix(knn_clf, 
        X_test, 
        y_test, 
        values_format='d', 
        display_labels=['No Stroke', 'Stroke'])


train_score = 0
test_score = 0
test_recall = 0
test_auc = 0

train_score = knn_clf.score(X_train, y_train)
test_score = knn_clf.score(X_test, y_test)
y_pred_knn = knn_clf.predict(X_test)

test_recall = recall_score(y_test, y_pred_knn)
knn_fpr, knn_tpr, thresholds = roc_curve(y_test, y_pred_knn)
test_auc = auc(knn_fpr, knn_tpr)
test_f1_score = f1_score(y_test,y_pred_knn)
test_precision_score = precision_score(y_test,y_pred_knn)



print("Train accuracy ", train_score)
print("Test accuracy ", test_score)
print("Test recall", test_recall)
print("Test AUC", test_auc)
print("Test f1 score", test_f1_score)
print("Test precision score", test_precision_score)

