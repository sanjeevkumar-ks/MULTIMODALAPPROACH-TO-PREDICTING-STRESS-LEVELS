# -*- coding: utf-8 -*-
"""
@author: sanjeev
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Comment this if the data visualisations doesn't work on your side
# %matplotlib inline

plt.style.use('bmh')

df = pd.read_csv('mexican_medical_students_mental_health_data.csv')
print(df.head())

print(df.info())
df = df.fillna(0)
# Add column to DataFrame using loc[]
df['PHQ9_Score'] = df.loc[:,['phq1', 'phq2','phq3','phq4','phq5','phq6','phq7','phq8','phq9']].sum(axis=1)
print(df)

# Add column to DataFrame using loc[]
df['gad_Score'] = df.loc[:,['gad1', 'gad2','gad3','gad4','gad5','gad6','gad7']].sum(axis=1)
print(df)

df['epw_Score'] = df.loc[:,['epw1', 'epw2','epw3','epw4','epw5','epw6','epw7','epw8']].sum(axis=1)
print(df)

df['stress_Score'] = df.loc[:,['PHQ9_Score', 'gad_Score','epw_Score']].mean(axis=1)
print(df)

# stress classification 0 = none, 1= mild,2=moderate,3=severe
PE_Conditions = [
    (df['stress_Score'] < 5),
    (df['stress_Score'] >= 5) & (df['stress_Score'] <= 9),
    (df['stress_Score'] >= 10) & (df['stress_Score'] <= 14),
    (df['stress_Score'] >= 15)
]
PE_Categories = [0, 1, 2, 3]
df['stress_Category'] = np.select(PE_Conditions, PE_Categories)
print(df)
print(df['thoughts_of_dropping_out'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['thoughts_of_dropping_out'], color='g', bins=100, hist_kws={'alpha': 0.4});

print(list(set(df.dtypes.tolist())))

df_num = df.select_dtypes(include = ['float64', 'int64','int32'])
print(df_num.head())

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations

# creating phq9 score

X = df_num.drop('stress_Category',axis=1)
y = df_num['stress_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
svc_model = SVC()
svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print("-----------------SVM classifier----------------------------")
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print("-----------------SVM with hyper parameter tuning classifier---------")
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))




clf_RF = RandomForestClassifier(n_estimators=25)
clf_RF.fit(X_train,y_train)
predictions1 = clf_RF.predict(X_test)
print("-----------------Random forest Classifier----------------------------")
print(confusion_matrix(y_test,predictions1))
print(classification_report(y_test,predictions1))


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("-----------------KNN Classifier----------------------------")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

lr =  LogisticRegression()
lr.fit(X_train,y_train)
y_pred1=lr.predict(X_test)

print("-----------------logistic regression Classifier----------------------")
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))

nb =  GaussianNB()
nb.fit(X_train, y_train)
y_pred2=nb.predict(X_test)
print("-----------------Naive Bayes Classifier--------------------")
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

