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

data = pd.read_csv("full.csv", low_memory=False)
empDfObj = pd.DataFrame(data, columns=['city', 'company', 'conditions', 'description', 'employment_mode', 'experience',
                                       'max_salary', 'min_salary', 'name', 'publish_date', 'requirements', 'responsibilities', 'schedule', 'skills', 'url', 'Класс'])
##Удаление вакансий у которых пропусков по признакам более 70%
# modObj = empDfObj.dropna(thresh=4)
# print(modObj.head())
le = preprocessing.LabelEncoder()

names = pd.read_csv("names.csv", low_memory=False, delimiter=';')
names_key = pd.DataFrame( columns=['№', 'Класс'])
names_key['Класс'] = names['Класс']
names_key['№'] = le.fit_transform(names['Класс'])

names_key.to_csv('encoded_names.csv',index=False)

empDfObj = empDfObj.apply(le.fit_transform)
##заменить на modObj

##Обучение ч2
X = empDfObj.iloc[:,:-1].values  
y = empDfObj['Класс'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)


logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
LR_prediction = logreg_clf.predict(X_test)
print(accuracy_score(y_test, LR_prediction))

SVC_model = SVC()    
SVC_model.fit(X_train, y_train) 
SVC_prediction = SVC_model.predict(X_test) 
print(accuracy_score(SVC_prediction, y_test))  

KNN_model = KNeighborsClassifier(n_neighbors=5)     
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(KNN_prediction, y_test))  


##ч2.1
print('g')