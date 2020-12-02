import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score  
from sklearn import preprocessing
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.linear_model import LogisticRegression 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv("full.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)
test = pd.DataFrame(test, columns=['city', 'company', 'conditions', 'description', 'employment_mode', 'experience',
                                       'max_salary', 'min_salary', 'name', 'publish_date', 'requirements', 'responsibilities', 'schedule', 'skills', 'url', 'Класс'])
train = pd.DataFrame(train, columns=['city', 'company', 'conditions', 'description', 'employment_mode', 'experience',
                                       'max_salary', 'min_salary', 'name', 'publish_date', 'requirements', 'responsibilities', 'schedule', 'skills', 'url', 'Класс'])

le = preprocessing.LabelEncoder()

train = train.apply(le.fit_transform)
test_n = test.apply(le.fit_transform)

X_train = train.iloc[:,:-1].values  
y_train = train['Класс'] 

X_test = test_n.iloc[:,:-1].values  


logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
LR_prediction = logreg_clf.predict(X_test)
output = test.copy()
output['Предполагаемый']=LR_prediction
output['class_key'] =  le.fit_transform(output['Класс'])
output[['name', 'Класс','class_key',  'Предполагаемый']].to_csv('output.csv',  mode = 'w' ,index=False)
print('g')
